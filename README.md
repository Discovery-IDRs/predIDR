# PredIDR

This is the repo for the PredIDR project in the Discovery IDR group.  The aim is to 1) comprehensively evaluate previous predictors of protein disorder and 2) build our own classifier to improve on past attempts.

## Project Organization

At the highest level, this project is organized into the following components:

```
PredIDR/
	├── bin/
	├── code/
	├── data/
	├── docs/
	|   └── file_management.md
	└── README.md
```

The top level directories are described below:
- `bin/`: Project "binaries," *i.e.* third-party programs or code which are used in this project, but not written by us. This directory will not be tracked by git, but our scripts may reference it by path, so it is included here for completeness.
- `code/`: Any code written by us.
- `data/`: Raw data used for our analysis. Consider this a "read-only" directory, meaning the data is never directly edited and the output of any analysis is stored in association with the code that produced it in `code/`. This directory will not be tracked by git, but our scripts may reference it by path, so it is included here for completeness.
- `docs/`: A place for notes and other pieces of documentation that are necessary for understanding the project.

This is only a broad overview, so if you plan on contributing to this project please read `file_management.md` in `docs/` which details the standards we have to work coherently as a team. Additional project files which are not appropriate for tracking by git can be found at our [Google Drive](https://drive.google.com/drive/folders/1h2HrEapw4jll0k-yVxKWsmqtmQxOabCZ?usp=sharing).
