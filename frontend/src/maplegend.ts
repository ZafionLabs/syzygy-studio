export const BODY_GRADIENTS: Record<string, string> = {
  earth: "linear-gradient(90deg,#000000,#1f4a7a,#2f8a7a,#5aa85a,#a8a855,#c8a87a,#fdfdfd)",
  moon: "linear-gradient(90deg,#00224e,#575d6d,#a59c74,#fee838)",
  mars: "linear-gradient(90deg,#000000,#7f4f32,#ff9f65,#ffc77f)",
};

export const LAYER_RAMPS: Record<string, string> = {
  heat: "linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4)",
  rain: "linear-gradient(90deg,#102018,#1a7a32,#c8e020,#f0a020,#e02020)",
  dust: "linear-gradient(90deg,#2a2118,#c47832,#f0c060,#fff4d0)",
  photo: "transparent",
};

function escapeHtml(value: string) {
  return value.replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char] ?? char));
}

export function mapLegendRow(input: {
  name: string;
  body: string;
  ramp?: string;
  low?: string;
  high?: string;
  available?: boolean;
}): string {
  const name = escapeHtml(input.name);
  if (input.available === false) {
    return `<div class="legend-row"><span class="name">${name} is not available for this body</span></div>`;
  }
  if (input.ramp === "photo") {
    return `<div class="legend-row"><span class="name">${name}</span></div>`;
  }
  const background = input.ramp && LAYER_RAMPS[input.ramp]
    ? LAYER_RAMPS[input.ramp]
    : (BODY_GRADIENTS[input.body] || BODY_GRADIENTS.moon);
  return `<div class="legend-row"><span class="name">${name}</span><span class="bar" style="background:${background}"></span><span>${escapeHtml(input.low || "")}</span><span>${escapeHtml(input.high || "")}</span></div>`;
}
