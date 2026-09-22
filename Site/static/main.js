const input = document.getElementById("input-area");
const button = document.getElementById("translate-button");
const output = document.getElementById("output-area");

async function translate() {
    const text = input.value.trim();
    if (!text) {
        output.textContent= "";
        return;
    }
    button.disabled = true;
    output.classList.remove("error")
    output.textContent = "Translating...";

    try {
        const response = await fetch("/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: text})
        });

        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`);
        }

        const result = await response.json();
        output.textContent = result.translation;
        console.log(result);
    } catch (error) {
        output.classList.add("error");
        output.textContent = "Translation failed. Please try again.";
        console.error(error.message);
    } finally {
        button.disabled = false;
    }
}

button.addEventListener("click", translate);
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        translate();
    }
});

