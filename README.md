# AgileProject

- Edita Buneviciene R00257694 email edita.buneviciene@mymtu.ie

- Edward Tracey R00257771 edward.tracey@mymtu.ie

- Dariusz Olczack Zwolinski R00242508 

- Patrick Strand R00064201 Email: p.strand@mymtu.ie


# Instructions

Make sure you have **git** installed on your machine. Also install **gitk**.

To clone this repository to your local machine use the command

    `git clone https://github.com/EditaBuneviciene/AgileProject`
    `cd AgileProject`

Type `git status` to see what branch you are on.

To create a new branch to make changes type `git branch <name_of_your_new_branch>`

To jump onto your new branch type `git checkout <name_of_your_new_branch>`

# Example:
  `git branch registration_menu`
  
  `git checkout registraion_menu`

The branch that you are on now is only local and does not exist on the git server yet.

To push your branch to the server use `git push -u origin <name_of_your_new_branch>`

It's best to commit changes you make in small chunks rather then just having one commit with a lot of changes.
To commit changes use the following command.

`git commit -a -m "A short discrtiption of your changes."`

# Example:

`git commit -a -m "Added menu dialog."`

`git commit -a -m "Added buttons."`

`git commit -a -m "Fixed bug with OK button."`


These commits are local. To push your commits to the servers copy of **<name_of_your_new_branch>** use 

`git push`

Once your are happy with all your changes you can merge the **master** branch into your code and test it.

`git merge master`

By doing it this way we ensure that any other changes that may have been made by other developers since you made your branch, can be picked up and tested with your code changes. 

This way **master** is always working.

Once you are happy with the changes you can merge your branch into **master**.

`git checkout master`

`git merge <name_of_your_new_branch>`




