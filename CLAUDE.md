# Rust DSP Effects Framework - Project Context

## Project Overview

**Goal**: Build a professional-grade audio effects framework in Rust that showcases DSP expertise, real-time performance, and embedded-ready architecture. This is a portfolio project targeting Digitech (audio effects company in Murray, UT) and the broader audio hardware industry.

**Timeline**: 3-4 weeks to "impressive" stage, then iterate publicly

**Primary deliverable**: Effects library + VST3 wrapper + comprehensive documentation

## Developer Background

**Morgan** - Audio engineer, Rust developer, music producer based in Salt Lake City
- Previously operated recording studio in St. George, UT
- Extensive audio engineering experience (mixing, mastering, production)
- Sample-based hip-hop and progressive house production
- Deep technical background: worked on OBELISK (chaos theory + embeddings project)
- Strong understanding of: dynamical systems, phase space, performance optimization
- Familiar with: faer, SIMD optimization, `no_std` constraints
- Some experience with nih-plug for VST development

## Core Design Principles

### 1. Pragmatic First, Elegant Second
- Every effect must match reference implementations (sound accuracy is paramount)
- Clean, maintainable code over clever abstractions
- Ship something impressive, don't chase mathematical beauty for its own sake
- Avoid "drift" - stay grounded in practical goals

### 2. Embedded-Ready Architecture
- Design everything to be `no_std` compatible from day one
- Target constraints: ~512KB RAM, strict CPU cycle budgets
- No allocations in audio processing paths
- Document hardware-ready design decisions even without physical hardware yet

### 3. Performance Targets
- Zero-allocation real-time processing
- SIMD optimization where applicable
- Lock-free parameter modulation
- Comprehensive benchmarks vs C/C++ implementations

### 4. Professional Quality
- Proper DSP algorithms (not shortcuts)
- Clean trait abstractions
- Extensive documentation
- Tests for correctness and performance

## Technical Stack

**Core Dependencies** (start here):
- `no_std` compatible from the start
- Consider: `dasp` for basic DSP primitives
- Consider: `nih-plug` for VST3 wrapper (Morgan has some experience)
- Consider: `fundsp` for reference/inspiration (but don't depend on it)

**Avoid**:
- Heavy allocations
- Complicated dependency chains
- Anything that can't run in `no_std` contexts

## Architecture Decisions

### Effect Trait Design
```rust
// Core abstraction - keep it clean and minimal
trait Effect {
    fn process(&mut self, input: f32) -> f32;
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);
    fn set_sample_rate(&mut self, sample_rate: f32);
    // Parameter handling - TBD
}
```

### Parameter System
- Lock-free updates (atomic operations)
- Smoothing to avoid zipper noise
- Clear mapping: normalized [0,1] ↔ physical units

### Memory Management
- All buffers pre-allocated
- No heap allocations in process() calls
- Use const generics where possible for compile-time sizing

## Effects Roadmap

**Phase 1: Core Framework (Week 1-2)**
- [ ] Core trait system
- [ ] Basic distortion effect (learn the patterns)
- [ ] Parameter handling with smoothing
- [ ] Unit tests for correctness
- [ ] Basic benchmarks

**Phase 2: Essential Effects (Week 3-4)**
- [ ] Chorus (modulation techniques)
- [ ] Delay (memory management patterns)
- [ ] Filter (IIR filter design)
- [ ] Reverb (if time permits - complex but impressive)

**Phase 3: Polish & Deployment**
- [ ] VST3 wrapper (using nih-plug)
- [ ] Professional README with audio examples
- [ ] Performance comparison documentation
- [ ] Publish first crates

## Effect Implementation Notes

### Distortion (Start Here)
- Simplest to implement correctly
- Focus: waveshaping, anti-aliasing (oversampling)
- Reference: Tube Screamer topology
- Good for establishing patterns

### Chorus
- Focus: LFO, modulated delay lines
- Reference: Classic analog chorus circuits
- Teaches interpolation and modulation

### Delay
- Focus: Circular buffers, feedback paths
- Reference: Analog delay pedals
- Critical for understanding memory management

### Filter
- Focus: Biquad filter design, coefficient calculation
- Reference: Moog ladder filter or state-variable filters
- Essential DSP primitive

## Code Style Preferences

**General**:
- Clear > clever
- Document the "why" not just the "what"
- Prefer explicit over implicit
- No premature optimization, but design for performance

**Naming**:
- Descriptive variable names
- Avoid abbreviations unless standard in DSP (e.g., `freq`, `q`, `gain`)
- Use `_hz`, `_db`, `_samples` suffixes for clarity

**Structure**:
- Flat module hierarchy initially
- Extract abstractions only when patterns become clear
- Keep related code together

**Testing**:
- Unit tests for mathematical correctness
- Integration tests for signal flow
- Benchmarks for performance validation
- Consider property-based testing for DSP invariants

## Performance Validation Strategy

**Benchmarks to include**:
- Single sample processing latency
- Block processing throughput
- Memory usage (stack only)
- SIMD vs scalar comparison
- vs equivalent C implementation (if available)

**Tools**:
- `criterion` for benchmarking
- `cargo bench` for regression tracking
- Consider flamegraphs for hotspot analysis

## Documentation Requirements

**README must include**:
- Clear project description and goals
- Audio examples (rendered output files)
- Usage examples with code
- Performance characteristics
- Embedded-ready callouts
- Roadmap for future work

**Each effect should document**:
- Algorithm overview
- Parameter ranges and meanings
- Computational complexity
- Reference implementations/papers
- Audio examples

## Future Considerations (After Initial Release)

**Nice-to-haves, but NOT for v1**:
- AI circuit modeling integration (save for later/separate project)
- Harmonic analysis framework (interesting but not core)
- Hardware implementation (when Daisy Seed budget available)
- Additional effects (compression, EQ, etc.)

**Keep these ideas in back pocket but DON'T pursue until core framework ships.**

## Project Portfolio Strategy

This project demonstrates:
1. **Deep DSP knowledge** - Classic effects implemented correctly
2. **Rust expertise** - Real-time safe, performant, embedded-ready
3. **Systems thinking** - Clean architecture, proper abstractions
4. **Execution** - Ships complete, documented, tested code

The goal is to show Digitech: "I can implement classic effects AND think about novel approaches" - framework first, innovation later.

## Key Reminders

- **Stay grounded**: Reference implementations are truth
- **Ship early**: 3-4 weeks to impressive, then iterate publicly  
- **Avoid drift**: Beautiful math is great IF it serves the practical goal
- **Document everything**: The write-up matters as much as the code
- **Benchmark religiously**: Performance claims need data

## Questions to Answer During Development

- How to handle different sample rates elegantly?
- Best way to expose parameters for VST wrapper?
- How much does SIMD actually help for single-sample processing?
- What's the right abstraction for modulation sources?
- How to make examples/tests sound good (not just correct)?

## Contact Context

If discussing with others:
- Digitech is in Murray, UT (right in Morgan's backyard)
- Portfolio targets audio hardware R&D roles
- Morgan brings unique combo: audio engineering + Rust + systems thinking
- Previous work on OBELISK demonstrates sophisticated technical thinking

---

**Last Updated**: [Date when starting project]
**Status**: Planning → Implementation
