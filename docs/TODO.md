List of documented issues that should be addressed

## Bugs
(none currently)

## Small Improvements
- Better dev setup for Claude (outline file locations, how to run/test)

## Features

### Substack Workflow (from VISION.md)

Stack a subset of frames and use the result as a brush source. This is the "killer feature" for fixing problem areas where the full stack produces artifacts.

**Use case:** Full stack has ghosting/mush in one area because frames 3-5 have motion blur. User:
1. Selects frames 1, 2, 6, 7 (excluding bad ones)
2. Stacks just those frames → "substack"
3. Uses substack as brush source to paint over the problem area in main result

**Implementation thoughts:**
- UI: Multi-select in sidebar (Ctrl+click or checkboxes)
- "Stack Selected" button creates a substack
- Substacks appear in source selector alongside original frames
- Could store multiple substacks per session
- Substacks are ephemeral (not saved to disk unless explicitly exported)

**Questions to resolve:**
- How to visually distinguish substacks from source frames in the list?
- Should substacks use same algorithm settings as main stack, or be configurable?
- Memory implications of holding multiple stacked results?