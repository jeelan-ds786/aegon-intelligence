import { describe, expect, it } from "vitest";

import { packageName } from "../src/index.js";

describe("@aegon/client-sdk", () => {
  it("loads the package", () => {
    expect(packageName).toBe("@aegon/client-sdk");
  });
});