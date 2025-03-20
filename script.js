const videoInput = document.getElementById("videoInput");
const uploadArea = document.getElementById("uploadArea");
const previewVideo = document.getElementById("previewVideo");
const analyzeButton = document.getElementById("analyzeButton");
const result = document.getElementById("result");
const uploadProgress = document.getElementById("uploadProgress");
const uploadProgressText = document.getElementById("uploadProgressText");
const analysisProgress = document.getElementById("analysisProgress");
const analysisProgressText = document.getElementById("analysisProgressText");

// 🎯 Handle click on upload area
uploadArea.addEventListener("click", () => {
    videoInput.click();
});

// 🎯 Handle file selection
videoInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

// 🎯 Drag & Drop Handling
uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = "white"; // Highlight the area
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.style.borderColor = "rgba(255, 255, 255, 0.5)";
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    videoInput.files = e.dataTransfer.files; // Update input file (some browsers may not support this)
    handleFile(file);
});

// 🎯 File validation & preview
function handleFile(file) {
    if (!file) return;

    if (file.type.startsWith("video/")) {
        const fileURL = URL.createObjectURL(file);
        previewVideo.src = fileURL;
        previewVideo.hidden = false;
        analyzeButton.hidden = false;
        result.textContent = "";
        uploadProgress.style.width = "0%";
        analysisProgress.style.width = "0%";
    } else {
        result.textContent = "⚠ Unsupported file format. Please upload a valid video file.";
        previewVideo.hidden = true;
        analyzeButton.hidden = true;
    }
}

// 🎯 Handle video analysis
analyzeButton.addEventListener("click", async () => {
    const file = videoInput.files[0];
    if (!file) {
        result.textContent = "⚠ No file selected.";
        return;
    }

    result.textContent = "Uploading...";
    uploadProgressText.hidden = false;
    uploadProgress.style.width = "0%";
    analyzeButton.disabled = true; // Prevent multiple clicks

    const formData = new FormData();
    formData.append("video", file);

    try {
        // 🚀 Simulated Upload Progress
        await fakeProgress(uploadProgress, uploadProgressText, "Uploading...", 1500);

        // 🚀 Upload Video
        let uploadResponse = await fetch("http://127.0.0.1:5000/upload", {
            method: "POST",
            body: formData
        });

        let uploadData = await uploadResponse.json();
        if (!uploadResponse.ok || !uploadData.success) {
            throw new Error(uploadData.error || "Upload failed.");
        }

        uploadProgress.style.width = "100%";
        uploadProgressText.textContent = "✅ Upload Complete!";
        uploadProgressText.hidden = true;

        // 🚀 Start Analysis
        analysisProgressText.hidden = false;
        analysisProgressText.textContent = "Analyzing...";
        
        await fakeProgress(analysisProgress, analysisProgressText, "Analyzing...", 2000);

        // 🎯 Call analyze function
        await analyzeVideo();
    } catch (error) {
        console.error("Upload Error:", error);
        result.textContent = "❌ Upload failed. Please try again.";
    } finally {
        analyzeButton.disabled = false; // Re-enable button
    }
});

// 🎯 Simulated progress animation
function fakeProgress(progressBar, progressText, message, duration) {
    return new Promise((resolve) => {
        let startTime = Date.now();
        let interval = setInterval(() => {
            let elapsed = Date.now() - startTime;
            let percentage = Math.min((elapsed / duration) * 100, 100);
            progressBar.style.width = percentage + "%";
            if (percentage >= 100) {
                clearInterval(interval);
                progressText.textContent = message + " ✅";
                resolve();
            }
        }, 100);
    });
}

// 🎯 Handle API call for analysis
async function analyzeVideo() {
    try {
        let analysisResponse = await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST"
        });

        let analysisData = await analysisResponse.json();
        if (!analysisResponse.ok || !analysisData.success) {
            throw new Error(analysisData.error || "Analysis failed.");
        }

        analysisProgress.style.width = "100%";
        analysisProgressText.textContent = "✅ Analysis Complete!";
        result.innerHTML = `<strong>🧐 Result:</strong> ${analysisData.result}`;
    } catch (error) {
        console.error("Analysis Error:", error);
        result.textContent = "❌ Error analyzing the video.";
    }
}
