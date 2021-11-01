# Optimization Algorithms Course - Student Workspace

This is a workspace for students to work on the Optimization Algorithms
Course coding assignments.

* Please fork this repo: Click the 'fork' button on top right of the gitlab webpage to create your own copy of this repo on gitlab.

* Clone your fork onto your computer, which includes the submodule that contains the actual assignments
```
git clone YOURFORK
cd oa-workspace
git submodule update --init --recursive
```

* Copy the specific assignment folder into your workspace, e.g.
```
cd oa-workspace
cp -R optimization_algorithms/assignments/a0_gradient_descent .
cd a0_gradient_descent
```

* Work on your solution. Test it with
```
python3 test.py
```

* Tell us the URL of your fork so that we can also evaluate it.

## How to fork if you don't have access to TUBerlin Gitlab

Create a repository in your personal Gitlab or Github and clone it to your computer

```
git clone YOURREPOSITORY
cd YOURREPOSITORY
```

Add our repository as remote of name upstream

```
git remote add upstream https://git.tu-berlin.de/lis-public/oa-workspace.git
```

Now you can fetch and pull from the upstream. Don't forget the submodule. 

```
git pull upstream main
git submodule update --init --recursive
```

Push to your own Gitlab/Github repository:

```
git push origin main
```
