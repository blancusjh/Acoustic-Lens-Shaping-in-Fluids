import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkPlane from "@kitware/vtk.js/Common/DataModel/Plane";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
window.vtk = {
  Rendering: {
    Misc: { vtkFullScreenRenderWindow },
    Core: { vtkMapper, vtkActor, vtkColorTransferFunction },
  },
  Common: { DataModel: { vtkPolyData, vtkPlane }, Core: { vtkDataArray } },
};
