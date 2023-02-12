# from flask import Blueprint, render_template, request, redirect, url_for, flash

# views = Blueprint(__name__,"views")

# @views.route("/")
# def home():
#     return render_template("index.html",name = "joe")


from flask import Flask, render_template, request, flash
from flask_cors import CORS


app = Flask(__name__,static_url_path='/static')
CORS(app)
app.secret_key = "manbearpig_MUDMAN888" ##

@app.route("/")
def index():
    flash("Enter a mail")
    return render_template("index.html")

@app.route("/mailController", methods=['POST', 'GET'])
def mailController():
	flash("SPAM " + str(request.form['mail']) + ", great to see you!")
	return render_template("index.html") 

if __name__ == '__main__':
    app.run()


    
