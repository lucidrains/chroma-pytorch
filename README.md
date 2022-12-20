<img src="./rfdiffusion.gif" width="450px"></img>

*generating a protein that binds to spike protein of coronavirus - Baker lab's concurrent RFDiffusion work*

## Chroma - Pytorch (wip)

Implementation of <a href="https://generatebiomedicines.com/chroma">Chroma</a>, generative model of proteins using DDPM and GNNs, in Pytorch. <a href="https://www.bakerlab.org/2022/11/30/diffusion-model-for-protein-design/">Concurrent work</a> seems to suggest we have a slight lift-off applying denoising diffusion probabilistic models to protein design.

<a href="https://stephanheijl.com/rfdiffusion.html">Explanation by Stephan Heijl</a>

If you are interested in open sourcing works like these out in the wild, please consider joining <a href="https://openbioml.org/">OpenBioML</a>

## Todo

- [ ] use <a href="https://huggingface.co/mrm8488/galactica-125m">galactica</a>

## Citations

```bibtex
@misc{
    title   = {Illuminating protein space with a programmable generative model},
    author  = {John Ingraham, Max Baranov, Zak Costello, Vincent Frappier, Ahmed Ismail, Shan Tie, Wujie Wang, Vincent Xue, Fritz Obermeyer, Andrew Beam, Gevorg Grigoryan},    
    year    = {2022},
    url     = {https://cdn.generatebiomedicines.com/assets/ingraham2022.pdf}
}
```
