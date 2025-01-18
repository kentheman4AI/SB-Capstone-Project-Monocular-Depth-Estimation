// Function to process each image
function processImage(doc) {
    // Duplicate the background layer
    var duplicatedLayer = doc.artLayers[0].duplicate();
    duplicatedLayer.name = "Resized Image";

    // Calculate the new size (1/6 of the original dimensions)
    var resizePercentage = 100 / 2;
    duplicatedLayer.resize(resizePercentage, resizePercentage);

    // Center the duplicated layer on the canvas
    var docWidth = doc.width.as('px');
    var docHeight = doc.height.as('px');
    var bounds = duplicatedLayer.bounds;
    var layerWidth = bounds[2].as('px') - bounds[0].as('px');
    var layerHeight = bounds[3].as('px') - bounds[1].as('px');
    var offsetX = (docWidth - layerWidth) / 2 - bounds[0].as('px');
    var offsetY = (docHeight - layerHeight) / 2 - bounds[1].as('px');
    duplicatedLayer.translate(offsetX, offsetY);

    // Select the bounds of the resized layer
    var selectionBounds = [
        [bounds[0].as('px'), bounds[1].as('px')], // Top-left
        [bounds[2].as('px'), bounds[1].as('px')], // Top-right
        [bounds[2].as('px'), bounds[3].as('px')], // Bottom-right
        [bounds[0].as('px'), bounds[3].as('px')]  // Bottom-left
    ];
    doc.selection.select(selectionBounds);

    // Stroke the selection to create a black border
    var borderThickness = 25; // Thickness of the border in pixels
    var borderColor = new SolidColor();
    borderColor.rgb.red = 50;
    borderColor.rgb.green = 0;
    borderColor.rgb.blue = 0;
    doc.selection.stroke(borderColor, borderThickness, StrokeLocation.OUTSIDE);

    // Deselect the selection
    doc.selection.deselect();
}

// Function to recursively process images in a folder and its subfolders
function processFolder(folder) {
    var files = folder.getFiles();
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        if (file instanceof Folder) {
            // Recursively process subfolders
            processFolder(file);
        } else if (file instanceof File && file.name.match(/\.(jpg|jpeg|png|tif|tiff|psd)$/i)) {
            // Open the image
            var doc = app.open(file);

            // Process the image
            processImage(doc);

            // Create a "processed" folder in the current directory
            var processedFolder = new Folder(folder + "/processed");
            if (!processedFolder.exists) {
                processedFolder.create();
            }

            // Save the processed file
            var outputFileName = file.name.replace(/\.[^\.]+$/, "") + "P.jpg";
            var saveOptions = new JPEGSaveOptions();
            saveOptions.quality = 12; // Maximum quality
            doc.saveAs(new File(processedFolder + "/" + outputFileName), saveOptions, true);

            // Close the document without saving changes
            doc.close(SaveOptions.DONOTSAVECHANGES);
        }
    }
}

// Main Script Wrapper
(function () {
    // Prompt user to select a folder
    var inputFolder = Folder.selectDialog("Select a folder with images to process:");
    if (!inputFolder) {
        alert("No folder selected. Script canceled.");
        return;
    }

    // Process the folder and its subfolders
    processFolder(inputFolder);

    alert("Processing completed. Files saved in 'processed' subfolders.");
})();