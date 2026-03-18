document.addEventListener('DOMContentLoaded', () => {
    const processBtn = document.getElementById('process-btn');
    const files = {}; // store files per card

    // Loop over 5 cards
    for (let c = 1; c <= 5; c++) {
        ['front', 'back', 'qr'].forEach(type => {
            const zone = document.getElementById(`drop-${type}-${c}`);
            const input = document.getElementById(`file-${type}-${c}`);
            const preview = document.getElementById(`preview-${type}-${c}`);

            if (!zone || !input || !preview) return;

            zone.onclick = () => input.click();

            zone.ondragover = (e) => {
                e.preventDefault();
                zone.classList.add('active');
            };
            zone.ondragleave = () => zone.classList.remove('active');
            zone.ondrop = (e) => {
                e.preventDefault();
                zone.classList.remove('active');
                if (e.dataTransfer.files.length) {
                    handleFile(c, type, e.dataTransfer.files[0], preview, zone);
                }
            };

            input.onchange = (e) => {
                if (input.files.length) {
                    handleFile(c, type, input.files[0], preview, zone);
                }
            };
        });
    }

    function handleFile(cardNum, type, file, preview, zone) {
        if (!file.type.startsWith('image/')) return;

        files[`card${cardNum}-${type}`] = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.style.backgroundImage = `url(${e.target.result})`;
            zone.classList.add('has-file');
            checkReady();
        };
        reader.readAsDataURL(file);
    }

    function checkReady() {
        // Enable button if at least 1 card has all 3 files
        for (let c = 1; c <= 5; c++) {
            if (files[`card${c}-front`] && files[`card${c}-back`] && files[`card${c}-qr`]) {
                processBtn.disabled = false;
                return;
            }
        }
        processBtn.disabled = true;
    }

    processBtn.onclick = async () => {
        processBtn.classList.add('loading');
        processBtn.disabled = true;

        const formData = new FormData();
        // append all uploaded files
        Object.keys(files).forEach(key => formData.append(key, files[key]));

        const layout = document.querySelector('input[name="layout"]:checked').value;
        formData.append('layout', layout);

        try {
            const response = await fetch('/process', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Processing failed');

            const html = await response.text();
            document.open();
            document.write(html);
            document.close();
        } catch (error) {
            alert('Error processing images. Please try again.');
            console.error(error);
            processBtn.classList.remove('loading');
            processBtn.disabled = false;
        }
    };
});
const layoutRadios = document.querySelectorAll('input[name="layout"]');
const cardsContainer = document.querySelector(".cards-container");

function updateLayout() {
    const selected = document.querySelector('input[name="layout"]:checked').value;

    if (selected === "document") {
        cardsContainer.classList.add("document-mode");
    } else {
        cardsContainer.classList.remove("document-mode");
    }
}

layoutRadios.forEach(radio => {
    radio.addEventListener("change", updateLayout);
});

// run when page loads
updateLayout();