# PredIDR

This is the repo for the PredIDR project in the Discovery IDR group.  The aim is to 1) comprehensively evaluate previous predictors of protein disorder and 2) build our own classifier to improve on past attempts.

## Project Organization

At the highest level, this project is organized into the following components:

```
PredIDR/
	├── analysis/
	├── bin/
	├── data/
	├── docs/
	|   └── file_management.md
        ├── src/
	└── README.md
```

The top level directories are described below:
- `analysis/`: Code and output files for scripts written by us to analyze data or accomplish a task in service of analyzing data. Any code within this directory is *ad hoc*, meaning it is "designed for a specific problem or task, non-generalizable, and not intended to be able to be adapted to other purposes." The majority of our project files will live here since our work will generally be answering a specific question about a specific data set rather than creating re-usable code that can be applied in multiple contexts.
- `bin/`: Project "binaries," *i.e.* third-party programs or code which are used in this project, but not written by us. This directory will not be tracked by git, but our scripts may reference it by path, so it is included here for completeness.
- `data/`: Raw data used for our analysis. Consider this a "read-only" directory, meaning the data is never edited and the output of any analysis is stored in association with the code that produced it in `code/`. This directory will not be tracked by git, but our scripts may reference it by path, so it is included here for completeness.
- `docs/`: A place for notes and other pieces of documentation that are necessary for understanding the project.
- `src/`: Re-usable code written by us. In practice, this will be modules of functions for tasks that are common across various parts of the project. We will package such code in `src/` both for convenience but also to provide a single source of truth for these core tasks. This project assumes that this directory is on the PYTHONPATH system variable, so modules within it are available for import statements.

This is only a broad overview, so if you plan on contributing to this project please read `file_management.md` in `docs/` which details the standards we have to work coherently as a team. Additional project files which are not appropriate for tracking by git can be found at our [Google Drive](https://drive.google.com/drive/folders/1h2HrEapw4jll0k-yVxKWsmqtmQxOabCZ?usp=sharing).
