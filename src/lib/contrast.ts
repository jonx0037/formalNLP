// WCAG 2.x contrast ratio between two #RRGGBB colors, used by the theme tests
// to keep Marginalia's accents legible (AA = 4.5:1 for body text) in both modes.
function relativeLuminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrastRatio(fgHex: string, bgHex: string): number {
  const [hi, lo] = [relativeLuminance(fgHex), relativeLuminance(bgHex)].sort((a, b) => b - a);
  return (hi + 0.05) / (lo + 0.05);
}
