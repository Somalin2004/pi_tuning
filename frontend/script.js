function predict() {

    let tau = parseFloat(document.getElementById("tau").value);
    let K = parseFloat(document.getElementById("K").value);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ tau: tau, K: K })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            "Kp: " + data.Kp + "<br>" +
            "Ki: " + data.Ki + "<br>" +
            "RL: " + data.RL_action;
    })
    .catch(error => console.log(error));
}