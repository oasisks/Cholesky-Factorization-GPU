# Cholesky-Factorization-GPU

Creating a Julia Wrapper to compute a high performance Cholesky Matrix Factorization algorithm on the GPU using a limited memory space.

## 10-Week Schedule

### Week 1:

- Read about the Cholesky Factorization algorithm and its use cases.
- Implement the Cholesky Factorization using Julia on the CPU.
- Benchmarking

### Week 2:

- Implement the naive implementation of the Cholesky Factorization Algorithm on GPU using Cuda.
- Bench Marking

### Week 3:

- Started reading the paper published by Jack Dongarra’s Group

### Week 4 – 5:

- Using Jack Dongarra’s algorithm, implement the Cholesky Factorization Algorithm in Julia using the GPU.
- Another set of benchmarking and a mini report (probably ½ - 1 page)
- Optimization is not a requirement currently.
- Midterm presentation
  o Introduction – what Cholesky Factorization is, the significance of it, and what we are doing.
  o Maybe a section about the general timeline (showing the schedule)
  o Showcasing research

### Week 6 - 9:

- Start thinking of ways to optimize the slow code.
- Find ways to decrease computation.
- Find ways to allocate memory to be pushed into the algorithm that overflows the memory limit.

### Week 10:

- Write a 1 – 2-page report of the findings.
  - Literature Review (short, maybe around 10 references)
  - Methodology and Algorithm
  - Compile all the benchmarking results.
  - Create a conclusion for the findings and the optimization that I did.
- End-of-term presentation created.
