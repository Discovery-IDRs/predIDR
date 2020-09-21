# Data Management

This is an overview of the file management strategy for this project.  The purpose is to have a system in place to both organize our work as it happens as well as to make it approachable to other current and future team members.

Each project is different and evolves over time and accordingly requires unique and flexible management standards. Thus, consider this a "living document" that is open to ongoing revision.

## Project Structure

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

Since these directories are described in the project README.md, this section will instead focus on the inner structure of the `analysis/` directory.

The `analysis/` directory contains code and output files for scripts written by us to analyze data or accomplish a task in service of analyzing data. We will use a two-tiered structure to organize our work. The first level is directories for major project goals. The second level is directories that contain scripts that accomplish specific tasks in pursuit of those larger goals. This project will likely have two level-one directories, one for our evaluation of other disorder predictors and one for creating our own:

```
analysis/
	├── evaluation_of_others/
	└── our_own_efforts/
```

These level-one directories will in turn contain folders of scripts that look something like this:


```
analysis/
	├── evaluation_of_others/
	|	├──pred1_eval/
	|	└──pred2_eval/
	└── our_own_efforts/
	 	├──neural_net_test1/
	 	└──neural_net_test2/
```

Finally, each of these level-two directories will contain scripts and their outputs. For simplicity, I'll only "expand" one.


```
analysis/
	├── evaluation_of_others/
	|	├──pred1_eval/
	|	└──pred2_eval/
	└── our_own_efforts/
	 	├──neural_net_test1/
	 	└──neural_net_test2/
	 	 	├──out/
	 	 	└──neural_net_test2.py
```

We'll save the output of any scripts to the conventionally-named `out/` directory. This is necessary in cases where a script produces a large number of files, but even if a script only produces a few files, saving them in their own directory prevents git from constantly harassing us about untracked files.

The specific names and numbers of level -one and -two directories are only a rough outline. As the semester progresses, we might find that one of these level-one goals will grow enough in size and complexity that we need to split it into two separate level-one directories. There are no hard and fast rules here. Project organization is less a set of prescriptions and more of a process of constantly asking, "Would someone with a reasonable background in data science who's never seen this project before be able to understand what's happening?"

## Naming Conventions

Directory and file names should be short and descriptive. Brevity is preferred at the highest levels of the project structure since they are accessed frequently. (There are also fewer directories and files here, so they are also inherently easier to remember). Descriptiveness is preferred at deeper levels where the contents are less familiar, and there are generally more files and folder to distinguish between.

Names should generally be in lowercase (excluding common or useful acronyms) and underscores must be in place of spaces between words or phrases.

Variable names in scripts follow the same guidelines. Where possible, use the conventional name for a variable if it is commonly used elsewhere in the project. However, consistency within one script or function is the most important.

## Code Style

We will broadly follow [PEP8](https://www.python.org/dev/peps/pep-0008/) conventions. To ensure our code is readable for a broad audience, use comments liberally and separate logical sections with blank lines.

## Documentation and Metadata



## Version Control

We are using GitHub to version our project and collaboratively write code using a fork and pull model. To contribute to this project, first fork the repository, make the changes and push them to your fork, and initiate a pull request to be reviewed by a team member before incorporating the modifications into the shared repo.

## Data and Storage

Raw data will be saved in the top level `data/` directory and the output of any analyses will be stored alongside the script that generates it in its `out/` directory. Data is considered "read-only" and thus will never be edited.

Data will not be versioned with git and will instead be stored on our [Google Drive](https://drive.google.com/drive/folders/1h2HrEapw4jll0k-yVxKWsmqtmQxOabCZ?usp=sharing). The directory structure of the Drive storage will exactly match that of the repository where data is output and stored on the local machines. No other files, however, will be stored in Drive.
