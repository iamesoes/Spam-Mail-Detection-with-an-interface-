from flask import Flask, render_template, request, flash
from flask_cors import CORS
from logistic_regression import predict_spam
from naive import predict_spam_2

app = Flask(__name__,static_url_path='/static')
CORS(app)
app.secret_key = "dnakjsm" ##


@app.route("/")
def index():
    flash("Enter a mail")
    return render_template("index.html")


@app.route("/mailController", methods=['POST', 'GET'])
def mailController():
    result, test_acc, train_acc = predict_spam([str(request.form["mail"])]) 
    flash(f"{result}  mail")
    return render_template("index.html", test_acc = test_acc, train_acc = train_acc) 


@app.route("/mailController_2", methods=['POST', 'GET'])
def mailController_2():
    result, test_acc, train_acc = predict_spam_2([str(request.form["mail"])]) 
    flash( result +" mail")
    return render_template("index.html", test_acc = test_acc, train_acc = train_acc) 


if __name__ == '__main__':
    app.run()
