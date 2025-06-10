# Git Workflow

We use a centralized Git workflow, meaning everyone clones the main repository on GitHub (assuming you have write permission), makes their edits in branches locally, and then makes pull requests for merger into the main branch.

## Naming Branches

The naming convention for branches is as follows (where `module` is the name of the module you are working on):

- **Feature Branches:** `module-feature-your-awesome-feature`
- **Bug Fix Branches:** `module-fix-your-bug-fix`

This helps in organizing the branches based on the module and the type of work being done, as well as helping other developers immediately identify the purpose of the branch. However, you do not have to follow this convention strictly if the situation calls for something that makes more sense. The most important thing is to be consistent and descriptive in your branch names.

## Workflow Steps

1. Create a new branch. This can be done on the website or locally. In general, this will be based on the latest version of the `main` branch.
2. Check out your new branch.
3. Make your changes in your local repository.
4. Commit your changes with a descriptive message.
5. Push your changes to the remote repository.
6. Open a pull request (PR) on GitHub to merge your changes into the `main` branch.

Someone who is tasked with reviewing PRs will review your changes and either approve them or request changes. Once approved, the PR can be merged into the `main` branch.

## Submitting a Pull Request (PR)

Once your work is ready and pushed, it is best to go to the GitHub repository in your browser. GitHub will prompt you to create a PR from your new branch to the `main` branch. Fill out the description, assign reviewers, and submit.

## Using the Command Line

The following commands are examples you can use to perform the actions listed above. If you are using a client like VS Code or GitHub Desktop, you can also perform these actions through the GUI.

**Stay Updated (Pull main into dev):**

```bash
git checkout dev
git pull origin main
git push origin dev
```

**Create a New Feature/Fix Branch:**

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-awesome-feature  # or fix/your-bug-fix
```

**Commit Your Changes:**

```bash
git status
git add .  # or git add <specific-files>
git commit -m "feat: Add awesome feature"  # Use conventional commit messages
```

**Push Your Branch:**

```bash
git push -u origin feature/your-awesome-feature
```

> **Note:** This requires write access to the `scottkleinman/uv_lexos` repository.

**Delete Local and Remote Branches (After Merge):**

```bash
git checkout dev
git pull origin dev
git branch -d feature/your-awesome-feature
git push origin --delete feature/your-awesome-feature
```
