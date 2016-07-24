MathJax.Hub.Config({
   "TeX" : {
       Macros: {
           gvn: "\\operatorname{|}",
           vct: ["\\boldsymbol{#1}", 1],
           mtx: ["\\mathbf{#1}", 1],
           set: ["\\mathcal{#1}", 1],
           fset: ["\\lbr #1 \\rbr", 1],
           reals: "\\mathbb{R}",
           lpa: "\\left(",
           rpa: "\\right)",
           lbr: "\\left\\lbrace",
           rbr: "\\right\\rbrace",
           lsb: "\\left[",
           rsb: "\\right]",
           prob: ["\\mathbb{P}\\lsb #1 \\rsb", 1],
           pden: ["\\mathbb{p}_{\\scriptscriptstyle #1}\\lsb #2 \\rsb", 2],
           rvar: ["\\mathsf{#1}", 1],
           rvct: ["\\bm{\\rvar{#1}}", 1],
           nrm: ["\\mathcal{N}\\lpa #1 \\rpa", 1],
           dr: "\\mathrm{d}",
           td: ["\\frac{\\dr #1}{\\dr #2}", 2],
           pd: ["\\frac{\\partial #1}{\\partial #2}", 2],
           pdd: ["\\frac{\\partial^2 #1}{\\partial #2 \\partial #3}", 3],
           tr: "^\\mathrm{T}",
       }
   }
});
