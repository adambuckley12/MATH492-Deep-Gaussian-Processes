-- expand-macros.lua
--
-- Expands custom \newcommand macros to standard LaTeX in every Math node.
-- Because we expand to standard LaTeX, the same .qmd source works for:
--   • HTML  – MathJax/KaTeX understands \mathbf, \mathbb, \mathcal, etc.
--   • PDF   – LaTeX compiles the standard commands directly
--   • Typst – Pandoc's built-in TeX→Typst converter handles the result
--
-- Macros are listed LONGEST-FIRST where names share a common prefix, so
-- that e.g. \vSigma is replaced before \vs is even considered.

-- ---------------------------------------------------------------------------
-- Zero-argument macros: { "name", "latex_replacement" }
-- ---------------------------------------------------------------------------
local zero_arg = {
  -- Greek bold matrices (longer names must come before shorter prefixes)
  { "vSigma",   "\\boldsymbol{\\Sigma}" },
  { "vLambda",  "\\boldsymbol{\\Lambda}" },
  { "vGamma",   "\\boldsymbol{\\Gamma}" },
  { "vTheta",   "\\boldsymbol{\\Theta}" },
  { "vOmega",   "\\boldsymbol{\\Omega}" },
  { "vPhi",     "\\boldsymbol{\\Phi}" },
  { "vPsi",     "\\boldsymbol{\\Psi}" },

  -- Greek bold vectors
  { "vepsilon", "\\boldsymbol{\\epsilon}" },
  { "vlambda",  "\\boldsymbol{\\lambda}" },
  { "vomega",   "\\boldsymbol{\\omega}" },
  { "vtheta",   "\\boldsymbol{\\theta}" },
  { "vsigma",   "\\boldsymbol{\\sigma}" },
  { "valpha",   "\\boldsymbol{\\alpha}" },
  { "vbeta",    "\\boldsymbol{\\beta}" },
  { "vgamma",   "\\boldsymbol{\\gamma}" },
  { "vdelta",   "\\boldsymbol{\\delta}" },
  { "vell",     "\\boldsymbol{\\ell}" },
  { "veta",     "\\boldsymbol{\\eta}" },
  { "vmu",      "\\boldsymbol{\\mu}" },
  { "vnu",      "\\boldsymbol{\\nu}" },
  { "vphi",     "\\boldsymbol{\\phi}" },
  { "vpsi",     "\\boldsymbol{\\psi}" },
  { "vtau",     "\\boldsymbol{\\tau}" },
  { "vxi",      "\\boldsymbol{\\xi}" },
  { "vzero",    "\\mathbf{0}" },

  -- Uppercase bold matrices
  { "vA", "\\mathbf{A}" }, { "vB", "\\mathbf{B}" }, { "vC", "\\mathbf{C}" },
  { "vD", "\\mathbf{D}" }, { "vF", "\\mathbf{F}" }, { "vG", "\\mathbf{G}" },
  { "vH", "\\mathbf{H}" }, { "vI", "\\mathbf{I}" }, { "vJ", "\\mathbf{J}" },
  { "vK", "\\mathbf{K}" }, { "vL", "\\mathbf{L}" }, { "vM", "\\mathbf{M}" },
  { "vP", "\\mathbf{P}" }, { "vQ", "\\mathbf{Q}" }, { "vR", "\\mathbf{R}" },
  { "vS", "\\mathbf{S}" }, { "vU", "\\mathbf{U}" }, { "vV", "\\mathbf{V}" },
  { "vW", "\\mathbf{W}" }, { "vX", "\\mathbf{X}" }, { "vY", "\\mathbf{Y}" },
  { "vZ", "\\mathbf{Z}" },

  -- Lowercase bold vectors
  { "vx", "\\mathbf{x}" }, { "vy", "\\mathbf{y}" }, { "vz", "\\mathbf{z}" },
  { "vf", "\\mathbf{f}" }, { "vw", "\\mathbf{w}" }, { "vu", "\\mathbf{u}" },
  { "vv", "\\mathbf{v}" }, { "va", "\\mathbf{a}" }, { "vb", "\\mathbf{b}" },
  { "vc", "\\mathbf{c}" }, { "vj", "\\mathbf{j}" }, { "vm", "\\mathbf{m}" },
  { "vn", "\\mathbf{n}" }, { "vp", "\\mathbf{p}" }, { "vq", "\\mathbf{q}" },
  { "vr", "\\mathbf{r}" }, { "vs", "\\mathbf{s}" }, { "vt", "\\mathbf{t}" },
  { "vk", "\\mathbf{k}" }, { "vh", "\\mathbf{h}" }, { "ve", "\\mathbf{e}" },

  -- Multi-letter operators (longer names before shorter prefixes)
  { "argmin", "\\operatorname{arg\\,min}" },
  { "argmax", "\\operatorname{arg\\,max}" },
  { "diag",   "\\operatorname{diag}" },
  { "defeq",  "\\coloneqq" },
  { "given",  "\\mid" },
  { "iid",    "\\stackrel{\\text{iid}}{\\sim}" },
  { "ind",    "\\stackrel{\\text{ind}}{\\sim}" },
  { "Var",    "\\operatorname{Var}" },
  { "Cov",    "\\operatorname{Cov}" },

  -- Two-letter / ambiguous – keep after longer names that share a prefix
  { "GP",  "\\mathcal{GP}" },
  { "KL",  "\\operatorname{KL}" },
  { "tr",  "\\operatorname{tr}" },

  -- Single-letter (last, so they never swallow the first char of a longer name)
  { "E", "\\mathbb{E}" },
  { "N", "\\mathcal{N}" },
  { "R", "\\mathbb{R}" },
}

-- ---------------------------------------------------------------------------
-- Two-argument macros: \name{arg1}{arg2}
-- (Uses a simple [^{}]* capture – sufficient for non-nested arguments.)
-- ---------------------------------------------------------------------------
local two_arg = {
  { "dd",  "\\frac{d\\,%1}{d\\,%2}" },
  { "pd",  "\\frac{\\partial\\,%1}{\\partial\\,%2}" },
  { "pdd", "\\frac{\\partial^{2}\\,%1}{\\partial\\,%2^{2}}" },
}

-- ---------------------------------------------------------------------------
-- Helper: replace \name when NOT followed by a letter (word-boundary safety)
-- ---------------------------------------------------------------------------
local function expand_zero(text, name, repl)
  -- Case 1: \name followed by a non-letter character
  text = text:gsub("\\" .. name .. "([^%a])", repl .. "%1")
  -- Case 2: \name at end of string
  text = text:gsub("\\" .. name .. "$", repl)
  return text
end

local function expand_two(text, name, repl)
  -- Matches \name{arg1}{arg2} with simple (non-nested) argument content
  text = text:gsub("\\" .. name .. "%s*{([^{}]*)}{([^{}]*)}", repl)
  return text
end

-- ---------------------------------------------------------------------------
-- Pandoc Math filter
-- ---------------------------------------------------------------------------
function Math(el)
  local t = el.text

  -- Expand two-arg macros first (so \pd isn't clobbered by a zero-arg \p)
  for _, m in ipairs(two_arg) do
    t = expand_two(t, m[1], m[2])
  end

  -- Expand zero-arg macros
  for _, m in ipairs(zero_arg) do
    t = expand_zero(t, m[1], m[2])
  end

  el.text = t
  return el
end
