-- One-time migration aid; the resulting LaTeX files can be edited directly.
local citations = {
  ['https://doi.org/10.1364/JOSAA.392795'] = 'silva2020',
  ['https://doi.org/10.1103/PhysRevE.106.065104'] = 'chesneau2022',
  ['https://doi.org/10.1121/1.5049579'] = 'bach2018',
  ['https://doi.org/10.1121/10.0005005'] = 'jorgensen2021',
  ['https://doi.org/10.1023/A:1011430410075'] = 'giles2000'
}
function Header(el)
  if el.level == 1 then return {} end
  el.level = el.level - 1
  -- LaTeX supplies section numbers.
  local label = pandoc.utils.stringify(el.content):gsub('^%d+%.%s*', '')
  el.content = pandoc.Inlines{pandoc.Str(label)}
  return el
end
function Link(el)
  local key = citations[el.target]
  if key then
    return pandoc.Inlines{pandoc.Span(el.content), pandoc.Space(),
      pandoc.RawInline('latex', '\\cite{' .. key .. '}')}
  end
  if not el.target:match('^https?://') then
    el.target = '../../docs/' .. el.target
  end
  return el
end
