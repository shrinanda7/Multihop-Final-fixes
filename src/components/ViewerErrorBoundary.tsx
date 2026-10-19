import React, { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
}

/** Keeps the chat panel mounted if the graph viewer throws (Cytoscape, layout, etc.). */
export class ViewerErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error("Graph viewer crashed without taking down the chat UI:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="flex-1 h-full flex items-center justify-center text-xs text-slate-500 px-6 text-center">
            Graph view could not be rendered. The answer panel is still available.
          </div>
        )
      );
    }
    return this.props.children;
  }
}
