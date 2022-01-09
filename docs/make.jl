pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add PotentialUQ to environment stack

using PotentialUQ
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(PotentialUQ, :DocTestSetup, :(using PotentialUQ); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "citations.bib"))

##
# Generate examples
##

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = Pair{String, String}[
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

examples = [title=>joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(bib;
    modules=[PotentialUQ],
    authors="CESMIX-MIT",
    repo="https://github.com/cesmix-mit/PotentialUQ.jl/blob/{commit}{path}#{line}",
    sitename="PotentialUQ.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cesmix-mit.github.io/PotentialUQ.jl",
        assets=String[],
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
        "API" => "api.md",
    ],
    doctest = true,
    linkcheck = true,
    strict = true,
)

deploydocs(;
    repo="github.com/cesmix-mit/PotentialUQ.jl",
    devbranch = "main",
    push_preview = true,
)