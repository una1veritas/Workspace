from flask import Flask, request, render_template_string

app = Flask(__name__)

# HTML content for the form
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Write File to Server</title>
</head>
<body>
    <h1>Save Data to a File</h1>
    <form action="/" method="POST">
        <textarea name="content" rows="10" cols="30" placeholder="Enter text to save..."></textarea><br>
        <button type="submit">Save to File</button>
    </form>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def write_file():
    if request.method == "POST":
        # Get the content from the form
        content = request.form.get("content", "")
        file_path = "output.txt"  # Specify the path to save the file

        # Save the content to a file
        try:
            with open(file_path, "w") as file:
                file.write(content)
            return "File written successfully!"
        except Exception as e:
            return f"Failed to write the file: {e}"

    # Render the form for GET requests
    return render_template_string(form_html)

if __name__ == "__main__":
    app.run(debug=True)