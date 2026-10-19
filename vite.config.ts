import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

function ignoreGeneratedWatchPath(watchPath: string) {
  const normalized = watchPath.replace(/\\/g, '/').toLowerCase();
  return (
    normalized.includes('/storage/') ||
    normalized.includes('/data/') ||
    normalized.includes('/uploads/') ||
    normalized.includes('/venv/') ||
    normalized.includes('/.venv/')
  );
}

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      // Query logs and extracted graphs are written during answer generation.
      // Watching those files causes a full reload that wipes the chat answer.
      watch: {
        ignored: ignoreGeneratedWatchPath,
      },
    },
  };
});
