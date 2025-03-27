# Neural Network Project

## Overview
This repository contains code for a neural network implementation that processes the MNIST dataset. The project includes different versions of the implementation, organized in separate folders (`V1`, `V2`, `V3`, `V4`).

## Folder Structure
```
/src
├── data/                # Contains MNIST dataset files
├── V1/                  # First version of the neural network
│   ├── nn.c             # Main neural network implementation
│   ├── makefile         # Makefile for compilation
│   ├── info.txt         # Additional information about this version
├── V2/                  # Second version of the neural network
│   ├── ...
├── V3/
│   ├── ...
├── V4/
│   ├── ...
└── README.md            # This file
```


Clone the Repository:
To start working on this project, clone the repository using:
git clone https://github.com/3astW1nd/hpc_proj.git


Build the Project:
cd V1  // Change to V1, or another version
make   // Compile the code


Run the Program:
After compilation, run the neural network program:
./nn.exe (For linux/WSL)
nn.exe (For windows)

MNIST Dataset:
The dataset files are stored in the `data/` folder. If the program cannot find them, update the file paths in `nn.c` to reflect the correct directory. 
// add a '../' to the file path in the code in V2, V3, V4 folders

How to use git:
1. Create a Branch
   // can make your own branch using the code below
   git checkout -b your-feature-branch
   // or you can just yolo and commit everything to the main branch (Recommended)
   
2. Make Changes & Commit
   git add .
   // might get a warning message about something something LF to CRLF next time git touches it
   // check troubleshooting 

   git commit -m "Your message here"
   
4. Push to GitHub
   git push origin your-feature-branch // if youre using seperate branches
   git push origin main // if youre using main branch (Recommended)

5. Pull from Github
   git pull origin main 
   

   
Troubleshooting:
Line Ending Issues: Run `git config --global core.autocrlf input`
  
