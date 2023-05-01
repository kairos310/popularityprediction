function postData(input) {
    $.ajax({
        type: "POST",
        url: "/test.py",
        data: { param: input },
        success: callbackFunc
    });
}

function callbackFunc(response) {
    // do something with the response
    console.log(response);
}

postData('data to process');