<?php
if ($_SERVER["REQUEST_METHOD"] === "POST") {
   // Convert the $_POST array to JSON format
   $jsonData = json_encode($_POST, JSON_PRETTY_PRINT);
   $content = $_POST['content'];
   $filePath = 'output.txt'; // Specify the path to save the file

    // Write the contents to the file
    if (file_put_contents($filePath, $jsonData)) {
        echo "File written successfully!";
    } else {
        echo "Failed to write the file.";
    }
} else {
    echo "Invalid request method.";
}
?>