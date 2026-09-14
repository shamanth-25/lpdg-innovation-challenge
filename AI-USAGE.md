# AI Usage

I used AI as a development assistant during this challenge.

## How I used AI

I used AI for:

- understanding the challenge requirements
- planning the project
- explaining Python and pandas code
- helping with data analysis
- suggesting different backtesting approaches
- checking the model logic
- analysing threshold results
- helping organize the project
- preparing documentation

I ran the code myself on the challenge data and checked the output locally.

The supplied `validate_submission.py` script was also used to verify the final predictions.

## One thing AI got wrong

During development, some of the early analysis focused mainly on hit rate.

The challenge, however, gives different costs for false positives and false negatives. I therefore changed the Part 2 analysis to compare the actual estimated cost at different thresholds instead of only comparing the number of hits.

The threshold analysis showed that 3.5 sigma had the lowest estimated cost among the tested thresholds.

## Another issue that was caught

The initial prediction code was written around the eight challenge weeks.

I identified that this would not be ideal for the live unseen-data evaluation. I changed the final predictor so that a prediction Monday and number of weeks can be supplied as arguments.

For example:

`python3 final_score.py --start-date 2026-04-06 --weeks 4`

This allows the same prediction logic to be used for a different prediction period.

## Verification

AI suggestions were treated as suggestions and were checked by running the code and comparing the results.

The final submission was validated locally using:

`python3 validate_submission.py predictions.csv`

The validator reported:

`predictions.csv: OK`

