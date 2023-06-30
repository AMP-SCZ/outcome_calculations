# outcome_calculations
In this repository we calculate the outcomes (predictors) for the AMP-SCZ study based on calculations provided by experts in the field.

### We calculate the outcomes for the AMP-SCZ study. 

Nora Penzel (March 30th, 2023)

All the outcomes are based on calculations or suggestions from experts
in the field, e.g., Jean Addington, Scott Woods, Dominic Oliver etc.
Outcomes should be used to predict endpoints.
In this repository we have 2 main scripts and four additional csv files.


### 1. outcome_calculations.py

This is the main script. Here all the outcome-calculations are implemented.
This script should be used in a two(three)-stage procedure.
The actual commands to run the script are given below
along with the explanation what happens depending on the input variables given:

1. First, it is recommended to run a test version.
   Especially after code has been changed and new updates have been made
   within the script.

    1a. To run it for Pronet:

        ./outcome_calculations.py "pronet" "test"

    1b. To run it for Prescient:

        ./outcome_calculations.py "prescient" "test"


   Thereby, we create 2 new csv files. These include all the outcomes based
   on the script for 6 subjects per network that have been previously checked.
   These csv files should then be used by the second script.


2. Run the second script `test_outcome.py`. If it displays that:

   *The data is the same as when we have checked it. Thus, go ahead and run the code*

   You can go back to `outcome_calculations.py` and run it with the following input variables:

    2a. To run it for Pronet:

        ./outcome_calculations.py "pronet" "run_outcome"

    2b. To run it for Prescient:

        ./outcome_calculations.py "prescient" "run_outcome"

    This now will create the outcomes within each subject folder.


**Developer mode**

3. If the function `test_outcome.py` produced that there was an error it should be further
   investigated what was different between the control files and the new files generated.
   In case code is updated and everything is checked the "control subject" should be
   re-generated running the code with:
   
        ./outcome_calculations.py "pronet" "create_check"
        ./outcome_calculations.py "prescient" "create_check"


4. After the creation of all outcomes etc. we create a dictionary using the R code
   `create_dictionary.R`. This R-code loads the csv files that are created when running
   the `outcome_calculations.py "run_outcome"` version. This script loads all the
   subject data and outcome data and provides overall scores. It should be run
   everytime the `outcome_calculations.py` was run to have the most up-to date
   values.


README beautified by Tashrif Billah.

