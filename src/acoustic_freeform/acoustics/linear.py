"""Exact element-interior elimination, not a reduced physical wave model."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu


class CondensedFactor:
    """Schur factorization for disjoint element-interior DOF blocks.

    Rejects inter-element interior couplings. Singular local blocks are not
    regularized: a caller must use the full factorization in that case.
    """

    def __init__(self, matrix, interior_dofs):
        self.size = matrix.shape[0]
        blocks = np.asarray(interior_dofs, dtype=int).T
        self.interior = blocks.ravel()
        if len(np.unique(self.interior)) != len(self.interior):
            raise ValueError("Interior degrees of freedom must be disjoint.")
        self.boundary = np.setdiff1d(np.arange(self.size), self.interior)
        self.inverse = None
        if not len(self.interior):
            self.factor = splu(matrix.tocsc(), permc_spec="MMD_AT_PLUS_A")
            return
        width = blocks.shape[1]
        rows_matrix = matrix.tocsr()
        ii = rows_matrix[self.interior][:, self.interior].tocoo()
        if np.any((ii.row // width != ii.col // width) & (ii.data != 0)):
            raise ValueError("Interior block contains inter-element couplings.")
        local = np.zeros((len(blocks), width, width), dtype=matrix.dtype)
        np.add.at(local, (ii.row // width, ii.row % width, ii.col % width), ii.data)
        inverse = np.linalg.inv(local)
        indices = np.arange(len(self.interior)).reshape((-1, width))
        rows = np.broadcast_to(indices[:, :, None], inverse.shape).ravel()
        cols = np.broadcast_to(indices[:, None, :], inverse.shape).ravel()
        self.inverse = coo_matrix(
            (inverse.ravel(), (rows, cols)), shape=(len(self.interior), len(self.interior))
        ).tocsr()
        self.ib = rows_matrix[self.interior][:, self.boundary]
        self.bi = rows_matrix[self.boundary][:, self.interior]
        schur = rows_matrix[self.boundary][:, self.boundary] - self.bi @ self.inverse @ self.ib
        del rows_matrix, ii, local, inverse, rows, cols
        self.factor = splu(schur.tocsc(), permc_spec="MMD_AT_PLUS_A")

    def solve(self, rhs):
        if self.inverse is None:
            return self.factor.solve(rhs)
        interior_rhs = self.inverse @ rhs[self.interior]
        boundary = self.factor.solve(rhs[self.boundary] - self.bi @ interior_rhs)
        solution = np.empty_like(rhs, dtype=np.result_type(rhs.dtype, self.inverse.dtype))
        solution[self.boundary] = boundary
        solution[self.interior] = interior_rhs - self.inverse @ (self.ib @ boundary)
        return solution


class ReusedCondensedFactor:
    """Reuse only a preconditioner; every GMRES solve uses the current matrix."""

    def __init__(self, matrix, interior_dofs):
        self.interior_dofs = interior_dofs
        self.matrix = matrix
        self.preconditioner = CondensedFactor(matrix, interior_dofs)
        self.iterative = False
        self.iterations = 0
        self.refreshes = 0

    @property
    def boundary(self):
        return self.preconditioner.boundary

    def update(self, matrix):
        if matrix.shape != self.matrix.shape:
            raise ValueError("Reused preconditioner requires the same pressure space.")
        self.matrix = matrix
        self.iterative = True
        self.iterations = 0

    def solve(self, rhs):
        if not self.iterative:
            return self.preconditioner.solve(rhs)
        vectors = rhs[:, None] if rhs.ndim == 1 else rhs
        result = np.empty_like(vectors, dtype=complex)

        def count(_):
            self.iterations += 1

        operator = LinearOperator(
            self.matrix.shape, matvec=self.preconditioner.solve, dtype=complex
        )
        for j, column in enumerate(vectors.T):
            answer, status = gmres(
                self.matrix,
                column,
                M=operator,
                rtol=1e-12,
                atol=0,
                restart=20,
                maxiter=3,
                callback=count,
                callback_type="pr_norm",
            )
            relative = np.linalg.norm(self.matrix @ answer - column) / max(
                np.linalg.norm(column), 1e-30
            )
            if status != 0 or not np.isfinite(relative) or relative > 2e-12:
                self.preconditioner = CondensedFactor(self.matrix, self.interior_dofs)
                self.refreshes += 1
                self.iterative = False
                return self.preconditioner.solve(rhs)
            result[:, j] = answer
        return result[:, 0] if rhs.ndim == 1 else result
