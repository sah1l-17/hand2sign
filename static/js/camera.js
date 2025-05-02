/**
 * Camera handling for sign language prediction page
 */

document.addEventListener('DOMContentLoaded', function() {
    // Only run on prediction page
    const videoElement = document.getElementById('videoElement');
    if (!videoElement) return;
    
    // Function to periodically update prediction
    function updatePrediction() {
        fetch('/get_prediction')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                const predictionResult = document.getElementById('predictionResult');
                const confidenceLevel = document.getElementById('confidenceLevel');
                const confidenceText = document.getElementById('confidenceText');
                
                if (data && data.label) {
                    // Update prediction display
                    predictionResult.textContent = data.label;
                    predictionResult.classList.add('bg-success', 'text-white');
                    setTimeout(() => {
                        predictionResult.classList.remove('bg-success', 'text-white');
                    }, 500);
                    
                    // Update confidence bar
                    const confidencePercent = Math.round(data.confidence * 100);
                    confidenceLevel.style.width = confidencePercent + '%';
                    confidenceText.textContent = confidencePercent + '%';
                    
                    // Change color based on confidence
                    if (confidencePercent > 90) {
                        confidenceLevel.style.backgroundColor = '#198754'; // green
                    } else if (confidencePercent > 70) {
                        confidenceLevel.style.backgroundColor = '#0d6efd'; // blue
                    } else {
                        confidenceLevel.style.backgroundColor = '#ffc107'; // yellow
                    }
                } else {
                    predictionResult.textContent = 'No sign detected';
                    confidenceLevel.style.width = '0%';
                    confidenceText.textContent = '0%';
                }
            })
            .catch(error => {
                console.error('Error fetching prediction:', error);
                document.getElementById('predictionResult').textContent = 'Error getting predictions';
            });
    }
    
    // Start polling for predictions
    const predictionInterval = setInterval(updatePrediction, 500);
    
    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        clearInterval(predictionInterval);
    });
    
    // Handle errors if the video feed fails
    videoElement.addEventListener('error', function(e) {
        console.error('Video element error:', e);
        handleMediaError({
            name: 'VideoError',
            message: 'Failed to load video feed'
        });
    });
    
    // Add a loading indicator until the video starts
    const loadingOverlay = document.createElement('div');
    loadingOverlay.id = 'videoLoadingOverlay';
    loadingOverlay.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">Connecting to camera...</p>
    `;
    loadingOverlay.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background-color: rgba(0,0,0,0.1);
        border-radius: 10px;
        z-index: 10;
    `;
    
    const videoContainer = videoElement.parentNode;
    videoContainer.style.position = 'relative';
    videoContainer.appendChild(loadingOverlay);
    
    // Remove loading overlay once video is playing
    videoElement.addEventListener('loadeddata', function() {
        const overlay = document.getElementById('videoLoadingOverlay');
        if (overlay) {
            overlay.remove();
        }
    });
});