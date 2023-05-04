/*
var W1 = [[-1.45686252, -0.34447819, -0.42561132, -0.3581678 , -1.65096112,
    -0.06997278,  1.33064607,  1.07879358,  0.31698702, -2.10262535,
    -0.44849363,  2.29527786, -1.59706437,  1.03076722, -0.49147108,
     0.94710884],
   [ 0.12383949, -1.22135449, -2.88233011,  0.04872792, -0.57607012,
     2.27095424, -0.02497977,  1.46254104,  0.07406453, -0.95779724,
     0.95580982,  0.3735373 ,  4.90899364, -1.90759822,  3.03032249,
    -0.88008291],
   [-1.5543388 ,  1.33108893,  2.24186641,  1.08179884, -2.93648451,
     0.48276441,  0.09804787,  0.61454688, -0.51063304, -4.39549752,
     1.09345236, -2.4391186 ,  0.50826068,  1.55274238, -1.88347541,
     0.60379689],
   [ 0.70661046, -0.31318132,  3.52683534, -0.01069675,  1.66883967,
    -1.03646886, -0.73296142, -1.66695726, -0.57253643,  2.66731384,
     0.03440808, -0.94877223, -1.97145114,  0.01000494, -1.54455662,
     0.17888923],
   [-0.5700786 , -1.46792009, -0.47247052, -0.01701051, -1.93550248,
    -0.16088828, -1.67331757,  0.92214624,  1.83699096, -3.63368136,
     1.69407573, -1.7058599 ,  1.36966512,  1.31596484, -0.95833354,
     2.52464783],
   [ 2.01506387,  0.64663984, -1.6920185 , -0.23939723, -0.29496782,
    -0.56486338, -2.35036696, -0.60008162, -0.41288028, -1.83268202,
     0.57593968, -1.29565303,  0.71953771,  0.11772044, -1.23934705,
     1.37051092],
   [ 0.06567653,  0.0813729 ,  0.21322706, -0.40569728, -4.22183823,
    -0.96644352, -0.38215478, -0.68506659,  1.52044913, -0.65232713,
     0.82063563,  0.61660828, -0.36664474, -0.58241763, -0.50246982,
    -1.11285271],
   [-0.60425222,  0.66689489,  0.12918495, -0.95471304, -1.17969869,
    -0.69469703, -0.51450332,  1.22756656, -1.30900814,  0.02902704,
     0.48149157, -0.87726623, -0.90235266, -0.84428681, -0.45354127,
    -3.77963034],
   [-1.05345358,  1.04620972, -1.111304  , -1.01129053,  0.04150735,
    -0.75581391,  0.7847405 , -0.15838525,  1.78117722, -0.62188322,
    -0.94157226, -1.79119269,  0.69797483,  2.93847864, -0.06718756,
     3.59935417]]
var W2 = [[ 1.83978997],
[ 2.05963707],
[ 2.6246346 ],
[ 1.28743343],
[ 1.87783756],
[ 3.10335081],
[ 1.34329494],
[ 2.52865257],
[ 2.62212002],
[-2.51456641],
[ 1.69028183],
[ 1.87442257],
[-2.79028659],
[ 2.87099275],
[ 1.86943531],
[-2.27499191]]

var b1 = [ 0.33489637,  1.51680384,  1.7581308 ,  0.7932608 , -1.90966653,
    2.97542132, -1.53489491,  1.67004015,  1.86084654, -4.65583544,
    1.1253406 , -0.09929488, -3.74057359,  2.10909353,  0.54744138,
   -2.8020827 ]

var b2 = [21.46638177]
*/

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
        console.log(this.W2)
        console.log(this.A1)
        this.Z2 = nj.dot(this.A1, this.W2).selection.data[0] + this.b2.selection.data[0]  // Linear transformation for the hidden layer
        return this.Z2  // Return the output without applying an activation function for regression
    }
}
let nn_model = new NeuralNet();

function predict_popularity(test_features){
    var mean =[ 0.631434, 0.598221  ,  -7.427778  ,   0.095441  , 0.28633117,   0.06100037,   0.1662222 ,   0.4901122 ,118.544022 ]
    var std = [ 0.14020394,  0.18692142,  2.82988386,  0.09608335,  0.27138501, 0.18647427,  0.11628525,  0.22731376, 28.8645816 ]
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


console.log(predict_popularity([0.8, 0.7, -5.0, 0.1, 0.1, 0.0, 0.1, 0.5, 120]))

function runmodel(e){
    console.log(e)
    var formData = document.getElementById("form");
    console.log(formData)
    form = document.getElementById("form").children
    inputs = []
    for(var i = 0; i < form.length; i++){
        if(i%4 ==  1){
            inputs.push(form[i].value)
        }
    }
    popularity = predict_popularity(inputs)
    document.getElementById("result").innerHTML = popularity
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