## Deploying the app onto Heroku


## Tutorial to use Heroku: [Cick here](https://www.kdnuggets.com/2021/04/deploy-machine-learning-models-to-web.html)	


## Progress:
* Deployed successfully.
* However, cannot process input:
    * Reason: the `qt platform plugin` cannot be initialized.
    * Error showed on log-file: `This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.`	

## What I am working on: (solution to the error above)
* Find an alternative to Heroku: pythonanywhere or replit, and see if it has the `qt platform plugin` error. If it does, the error may come from conflict of python and opencv's versions.
