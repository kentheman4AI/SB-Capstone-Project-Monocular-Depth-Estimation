// Function to process a single image
function processImage(doc) {
    var width = doc.width.as('px');
    var height = doc.height.as('px');

    // Calculate the area to copy from the left half (1/9th of the image)
    var leftHalfWidth = width / 2;
    var selectionWidth = leftHalfWidth / 2; // 1/9th of the entire image width
    var selectionHeight = height / 2; // 1/9th of the entire image height
    var x1 = (leftHalfWidth - selectionWidth) / 2; // Centered on the left half
    var y1 = (height - selectionHeight) / 2; // Centered vertically
    var x2 = x1 + selectionWidth;
    var y2 = y1 + selectionHeight;

    // Select and copy the region
    doc.selection.select([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]);
    doc.selection.copy();

    // Deselect the selection
    doc.selection.deselect();

    // Paste the copied region
    doc.paste();
    var pastedLayer = doc.activeLayer;

    // Flip the pasted layer horizontally to create a mirror image
    pastedLayer.resize(-100, 100); // Flip horizontally by inverting the horizontal scale

    // Move the pasted layer to the right half, centered
    var rightHalfCenterX = (leftHalfWidth + width) / 2;
    var pastedBounds = pastedLayer.bounds;
    var pastedWidth = pastedBounds[2].as('px') - pastedBounds[0].as('px');
    var pastedHeight = pastedBounds[3].as('px') - pastedBounds[1].as('px');
    pastedLayer.translate(
        rightHalfCenterX - (pastedBounds[0].as('px') + pastedWidth / 2),
        height / 2 - (pastedBounds[1].as('px') + pastedHeight / 2)
    );

    // Draw a silver border around the pasted region
    var silverColor = new SolidColor();
    silverColor.rgb.red = 192; // RGB for silver
    silverColor.rgb.green = 192;
    silverColor.rgb.blue = 192;
    doc.selection.select([
        [pastedBounds[0].as('px') + pastedWidth, pastedBounds[1].as('px')],
        [pastedBounds[2].as('px') + pastedWidth, pastedBounds[1].as('px')],
        [pastedBounds[2].as('px') + pastedWidth, pastedBounds[3].as('px')],
        [pastedBounds[0].as('px') + pastedWidth, pastedBounds[3].as('px')]
    ]);
    doc.selection.stroke(silverColor, 30, StrokeLocation.OUTSIDE); // Border thickness = 30px
    doc.selection.deselect();

    // Rotate the pasted layer by 20 degrees
    //pastedLayer.rotate(20);

    // Merge the layers (optional, comment this line if merging is not needed)
    doc.mergeVisibleLayers();
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

            // Create a "processed" folder in the top-level directory
            var processedFolder = new Folder(folder + "/processed");
            if (!processedFolder.exists) {
                processedFolder.create();
            }

            // Save the processed file with "_P" suffix
            var originalName = file.name.replace(/\.[^\.]+$/, ""); // Remove the file extension
            var outputFileName = originalName + "_P.jpg"; // Add "_P" suffix
            var saveOptions = new JPEGSaveOptions();
            saveOptions.quality = 12; // Maximum quality
            doc.saveAs(new File(processedFolder + "/" + outputFileName), saveOptions, true);

            // Close the document without saving changes
            doc.close(SaveOptions.DONOTSAVECHANGES);

            // Delete the original file after processing
            if (file.exists) {
                file.remove();
            }
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