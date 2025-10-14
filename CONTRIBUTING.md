# Recommended steps for contributors

Contributors to `NuLattice` should follow the general steps of:
1. [Forking](https://github.com/NuLattice/NuLattice/fork) the repository;
2. Adding new developments, fixes, optimizations, or other contributions in a new branch (not `main`);
3. And merging their contributions back to the main repository via a [pull request](https://github.com/NuLattice/NuLattice/pulls).

Below we provide instructions on how contributors can do this step by step, including instructions on making new developments private until they are ready to be made public and merged back into the main repository.

## Forking the repository

To fork `NuLattice`, simply click the button on the top right of [the main repository page](https://github.com/NuLattice/NuLattice):

![Forking NuLattice](contributing/forking.png "How to fork NuLattice")

The button is highlighted in red. This creates a new repository forked from `NuLattice` at the url `https://github.com/[your Github username]/NuLattice`. You can modify the fork however you like, including adding collaborators, and you will use the fork to contribute your developments and improvements back to `NuLattice` once they are ready (see below).

Now that you have a fork of `NuLattice`, simply clone the remote repository:
```
git clone https://github.com/[your Github username]/NuLattice
```
to obtain a local repository that you can develop.

## Developing a branch

We strongly encourage contributors to create new branches for their work. Creating a branch to develop a new feature allows the `main` branch of your fork to be easily synchronized with the main `NuLattice` `main` branch. This can simplify the process of merging back to the main `NuLattice` repository significantly.

To work with a new branch for a feature, create the branch in your local repository:
```
git checkout -b [feature branch name]
```
and push the branch to the Github remote:
```
git push --set-upstream origin [feature branch name]
```

You can then develop new features for `NuLattice` following standard practices. If you are unfamiliar with `git`, consider reading the [Github git basics guide](https://github.com/git-guides#learning-git-basics).

