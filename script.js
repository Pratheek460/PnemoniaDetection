async function predict() {

    const file =
        document.getElementById("image").files[0];

    if (!file) {
        alert("Please select an X-Ray image");
        return;
    }

    let formData = new FormData();

    formData.append(
        "file",
        file
    );

    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/predict?patient_name=Pratheek",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        document.getElementById("result").innerHTML = `

            <h2>
                Prediction: ${data.prediction}
            </h2>

            <h3>
                Confidence: ${data.confidence}%
            </h3>

            <br>

            <a
                href="http://127.0.0.1:8000/report/${data.id}"
                target="_blank"
            >
                Download Report
            </a>

            <br><br>

            <h3>
                Grad-CAM Heatmap
            </h3>

            <img
                src="http://127.0.0.1:8000/heatmap/${data.heatmap.split('/').pop()}"
                width="400"
            />

        `;

    } catch (error) {

        console.error(error);

        document.getElementById("result").innerHTML = `
            <h3>Error occurred while predicting.</h3>
        `;
    }
}