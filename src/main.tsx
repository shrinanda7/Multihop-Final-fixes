import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import { ViewerErrorBoundary } from './components/ViewerErrorBoundary.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <ViewerErrorBoundary
    fallback={
      <div style={{ height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, color: "#9898b0", padding: "0 24px", textAlign: "center" }}>
        The interface hit an error. Reload the page — your last answer is kept in this tab if one was generated.
      </div>
    }
  >
    <App />
  </ViewerErrorBoundary>
);

