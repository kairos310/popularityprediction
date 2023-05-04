class NeuralNet{
    constructor(){
        this.W1 = nj.array(W1)  // Initialize weights for the input layer
        this.b1 = nj.array([b1]) // Initialize biases for the input layer
        this.W2 = nj.array(W2) // Initialize weights for the hidden layer
        this.b2 = nj.array(b2)  // Initialize biases for the hidden layer
    }
    forward(X){
        this.Z1 = nj.add(nj.dot(X, this.W1), this.b1)  // Linear transformation for the input layer
        this.A1 = relu(this.Z1)  // Apply ReLU activation function to the input layer
        this.Z2 = nj.dot(this.A1, this.W2).selection.data[0] + this.b2.selection.data[0]  // Linear transformation for the hidden layer
        return this.Z2  // Return the output without applying an activation function for regression
    }
}
let nn_model = new NeuralNet();

function predict_popularity(test_features){
    var test_features_normalized = [];
    for(var i = 0; i < mean.length; i++){
        test_features_normalized[i]  = (test_features[i] - mean[i]) / std[i]
    }
    var nn_prediction = nn_model.forward(nj.array([test_features_normalized]))
    return nn_prediction
}


function relu(x){
    //console.log(x.selection.data)
    var ret = []
    for(var i = 0; i < x.selection.data.length; i++){
        ret[i] = nj.max(x.selection.data[i], 0)
    }
    return nj.array([ret])
}

function runmodel(e){
    var formData = document.getElementById("form");
    form = document.getElementById("form").children
    inputs = []
    for(var i = 0; i < form.length; i++){
        if(i%4 ==  1){
            inputs.push(form[i].value)
        }
    }
    popularity = predict_popularity(inputs)
    if(popularity > 100){
        popularity = 100 + " model(" + popularity + ")";
    }else if(popularity < 0){
        popularity = 0 + " model(" + popularity + ")"
    }
    document.getElementById("result").innerHTML = "Popularity: " + popularity
}


document.addEventListener("input", (event)=>{runmodel(event)})
/*  

//HTML REQUESTS
const form = document.getElementById("form");
const submitter = document.querySelector("button[value='Run Algorithm']");
const formData = new FormData(form, submitter);

const output = document.getElementById("output");

for (const [key, value] of formData) {
  output.textContent += `${key}: ${value}\n`;
}
*/