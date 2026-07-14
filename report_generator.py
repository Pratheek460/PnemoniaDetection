from reportlab.platypus import *

def generate_report(
    patient,
    prediction,
    confidence,
    report_path
):

    doc = SimpleDocTemplate(
        report_path
    )

    content=[]

    content.append(
        Paragraph(
            "PNEUMONIA REPORT"
        )
    )

    content.append(
        Paragraph(
            f"Patient: {patient}"
        )
    )

    content.append(
        Paragraph(
            f"Prediction: {prediction}"
        )
    )

    content.append(
        Paragraph(
            f"Confidence: {confidence}%"
        )
    )

    doc.build(content)