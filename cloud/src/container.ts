import { Container } from "@cloudflare/containers";

export class MarkerContainer extends Container {
  defaultPort = 8080;
  sleepAfter = "5m";

  override onStart(): void {
    console.log("Marker container started");
  }

  override onStop(): void {
    console.log("Marker container stopped");
  }

  override onError(error: unknown): void {
    console.error("Marker container error:", error);
  }
}
