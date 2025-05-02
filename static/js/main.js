/**
 * Main JavaScript file for Sign Language Recognition Web App
 */

document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to main content
    document.querySelector('main').classList.add('fade-in');
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            if (!targetId) return;
            
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Enable tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Add active class to current page in navbar
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentLocation) {
            link.classList.add('active');
        }
    });
    
    // Automatically collapse navbar on mobile after clicking a link
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
        document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
            link.addEventListener('click', () => {
                if (window.getComputedStyle(navbarToggler).display !== 'none') {
                    navbarCollapse.classList.remove('show');
                }
            });
        });
    }
});

// Function to show an alert for camera permission
function showCameraPermissionAlert() {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-warning alert-dismissible fade show';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        <strong>Camera Access Required!</strong> Please allow camera access to use the sign language recognition feature.
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Insert at the top of the main content
    const mainContent = document.querySelector('main');
    if (mainContent && mainContent.firstChild) {
        mainContent.insertBefore(alertDiv, mainContent.firstChild);
    }
}

// Function to handle media errors
function handleMediaError(error) {
    console.error('Media error:', error);
    
    let errorMessage = 'An error occurred accessing the camera.';
    
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        errorMessage = 'Camera access was denied. Please allow camera access to use this feature.';
        showCameraPermissionAlert();
    } else if (error.name === 'NotFoundError') {
        errorMessage = 'No camera was found on your device.';
    } else if (error.name === 'NotReadableError') {
        errorMessage = 'Your camera might be in use by another application.';
    }
    
    // Display error message
    const videoElement = document.getElementById('videoElement');
    if (videoElement) {
        videoElement.style.display = 'none';
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger text-center p-5';
        errorDiv.textContent = errorMessage;
        videoElement.parentNode.appendChild(errorDiv);
    }
}