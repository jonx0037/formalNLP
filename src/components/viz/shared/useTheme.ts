import { useEffect, useState } from 'react';
import type { Theme } from './colorScales';

// Tracks <html data-theme>, which ThemeToggle and the BaseLayout bootstrap set,
// so D3 graphs can redraw in the right palette when the reader toggles themes.
export function useTheme(): Theme {
  const read = (): Theme =>
    typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'dark'
      ? 'dark'
      : 'light';
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(read()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    setTheme(read());
    return () => observer.disconnect();
  }, []);
  return theme;
}
