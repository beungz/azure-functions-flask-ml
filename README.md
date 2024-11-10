### Team Assignment 9
***
**Serverless Deployment using Azure Functions**<br>
**Team**: Matana Pornluanprasert (Matana P)<br>

This is an Income Prediction ML model deployment using Azure Functions, where the ML model was created by Hoshang Karnjekar (hoshangk).
The front end is based on Flask Framework with CSS stylesheet.
<br>
<br>


Requirements:<br>
```
azure-functions
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.5.2
flask==3.0.3
```
<br>

#### **How to run the code locally, with Azure Functions extension and Azure Functions Core Tools**
***
To run the code, go to the main directory of this code, and type the followings in the terminal<br>

```
func start
```

Azure Functions Core Tools will return the URL for local browser<br>

```
Functions:

        HttpTrigger1: [GET,POST] http://localhost:7071/{*route}
```

Open your local browser, go to the URL above `http://localhost:7071`, key in your inputs, then click submit button.

You should see output like this.

```
Predicted income: Income more than 50K 
```
