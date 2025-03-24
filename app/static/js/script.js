document.addEventListener('DOMContentLoaded', function() {
    const poemForm = document.getElementById('poem-form');
    const generateBtn = document.getElementById('generate-btn');
    const resultContainer = document.getElementById('result-container');
    const poemResult = document.getElementById('poem-result');
    const copyBtn = document.getElementById('copy-btn');

    poemForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Generating...';
        resultContainer.style.display = 'none';
        
        try {
            const formData = new FormData(poemForm);
            
            const response = await fetch('/generate-poem', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Display the result
            poemResult.textContent = data.poem;
            resultContainer.style.display = 'block';
            
            // Scroll to result
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } catch (error) {
            console.error('Error:', error);
            poemResult.textContent = 'An error occurred while generating the poem. Please try again.';
            resultContainer.style.display = 'block';
        } finally {
            // Reset button state
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Generate Poem';
        }
    });
    
    // Copy poem to clipboard
    copyBtn.addEventListener('click', function() {
        const poemText = poemResult.textContent;
        navigator.clipboard.writeText(poemText).then(function() {
            // Change button text temporarily
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            
            setTimeout(function() {
                copyBtn.innerHTML = originalText;
            }, 2000);
        }).catch(function(err) {
            console.error('Could not copy text: ', err);
        });
    });
});
