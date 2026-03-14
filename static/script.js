document.addEventListener('DOMContentLoaded', () => {
    const drops = ['front', 'back', 'qr'];
    const files = { front: null, back: null, qr: null };
    const processBtn = document.getElementById('process-btn');
    const resultsSection = document.getElementById('results-section');

    drops.forEach(type => {
        const zone = document.getElementById(`drop-${type}`);
        const input = document.getElementById(`file-${type}`);
        const preview = document.getElementById(`preview-${type}`);

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
                handleFile(type, e.dataTransfer.files[0], preview, zone);
            }
        };

        input.onchange = (e) => {
            if (input.files.length) {
                handleFile(type, input.files[0], preview, zone);
            }
        };
    });

    function handleFile(type, file, preview, zone) {
        if (!file.type.startsWith('image/')) return;

        files[type] = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.style.backgroundImage = `url(${e.target.result})`;
            zone.classList.add('has-file');
            checkReady();
        };
        reader.readAsDataURL(file);
    }

    function checkReady() {
        if (files.front && files.back && files.qr) {
            processBtn.disabled = false;
        }
    }

    processBtn.onclick = async () => {
        processBtn.classList.add('loading');
        processBtn.disabled = true;

        const formData = new FormData();
        formData.append('front', files.front);
        formData.append('back', files.back);
        formData.append('qr', files.qr);

        const layout = document.querySelector('input[name="layout"]:checked').value;
        formData.append('layout', layout);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

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
