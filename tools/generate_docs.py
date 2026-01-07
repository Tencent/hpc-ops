from pathlib import Path
import ast

if __name__ == "__main__":
    # list funcs in hpc
    pydir = Path(__file__).parent.parent.joinpath("hpc")

    file_funcs = []
    for f in pydir.glob("*.py"):
        if f.name.startswith("__"):
            continue

        file_func = {"file": f.stem, "funcs": []}

        with open(f, "r", encoding="utf-8") as fp:
            code = fp.read()

        tree = ast.parse(code, filename=f.name)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                file_func["funcs"].append(node.name)
        file_funcs.append(file_func)

    # generate docs for each file
    docsdir = Path(__file__).parent.parent.joinpath("docs")

    for file_func in file_funcs:
        file = docsdir.joinpath(file_func["file"]).with_suffix(".md")
        with open(file, "w") as fp:
            for func in file_func["funcs"]:
                text = "::: hpc.{}.{}\n".format(file_func["file"], func)
                fp.write(text)

    # copy readme as index
    readme = Path(__file__).parent.parent.joinpath("README.md")
    index = Path(__file__).parent.parent.joinpath("docs/index.md")

    with open(readme, "r", encoding="utf-8") as fp:
        data = fp.read()

    with open(index, "w", encoding="utf-8") as fp:
        fp.write(data)
