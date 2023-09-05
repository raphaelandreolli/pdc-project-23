# Project for the PDC summer school 2023

Project by Axel Lundkvist, Raphael Andreolli, Vinit Nagda.

## 1 Description
In this project, the task is to work in groups of 1-3 students and parallelize the
”energy storms” code by using OpenMP, MPI, and CUDA/HIP. In addition,
the group must write a short report (maximum five pages in the standard latex
template) where they describe the techniques they used and reason around
possible future opportunities for parallelization.

The energy storms project was developed by Arturo Gonzalez-Escribano and
Eduardo Rodriguez-Gutiez in Group Trasgo, Universidad de Valladolid (Spain)
and a handout with a more detailed description of the software is provided as a
separate document.

## 2 Formal requirements
For a passing grade, the group must do the following.
- Develop one version of the code parallelized with OpenMP.
- One version parallelized with MPI.
- One version running on a GPU, using either the HIP or CUDA programming
language.
- Verify that the codes are correct and able to reproduce the results from
the sequential baseline.
- Compare the different versions with regard to run time and parallelization
strategy.
- Reason around possible future optimization strategies.
- Hand in a report by the 6th of September 23:59 AOE, detailing the work
and providing a link to a repository with the code and clear instructions on
how to compile and run it. The final report and a link to the code should
be sent by email addressed to makarp@kth.se and markidis@kth.se.
