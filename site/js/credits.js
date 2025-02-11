function init() {
  function add_model_card(
    container,
    model_name,
    authors,
    paper_link,
    source_link,
    body,
    paper_abstract,
    other = ""
  ) {
    let card_html = `
            <div class="card border-secondary mb-3">
              <div class="card-body">
                  <h4 class="card-title">${model_name}</h4>
                  <h6 class="card-subtitle mb-2 text-muted">
                    ${authors}
                    <br>
                  </h6>
                  <p class="card-text text-body-secondary">
                    ${body}
                    </p>
              <h6>Abstract</h6>
              <p class="card-text text-primary-emphasis">
                ${paper_abstract}
              </p>

              </div>
                <div class="card-footer text-muted">
                  <a href="${paper_link}" class="card-link">Paper</a>
                  <a href="${source_link}" class="card-link">Source</a>
                  ${other}
              </div>
            </div>
                    `;
    container.appendChild(
      new DOMParser().parseFromString(card_html, "text/html").body.firstChild
    );
  }

  const model_card_container = document.getElementById("model-card-container");
  add_model_card(
    model_card_container,
    "Whisper SpecRNet",
    "Piotr Kawa, Marcin Plata, Michał Czuba, Piotr Szymański, Piotr Syga",
    "https://www.isca-speech.org/archive/interspeech_2023/kawa23b_interspeech.html",
    "https://www.github.com/piotrkawa/deepfake-whisper-features/tree/main",
    "",
    // ` A model trained using features extracted from <a href="https://openai.com/index/whisper/">Whisper</a>`,
    `With a recent influx of voice generation methods, the threat introduced by audio DeepFake (DF) is ever-increasing. Several different detection methods have been presented as a countermeasure. Many methods are based on so-called front-ends, which, by transforming the raw audio, emphasize features crucial for assessing the genuineness of the audio sample. Our contribution contains investigating the influence of the state-of-the-art <a href="https://openai.com/index/whisper/">Whisper</a> automatic speech recognition model as a DF detection front-end. We compare various combinations of Whisper and well-established front-ends by training 3 detection models (LCNN, SpecRNet, and MesoNet) on a widely used ASVspoof 2021 DF dataset and later evaluating them on the DF In-The-Wild dataset. We show that using Whisper-based features improves the detection for each model and outperforms recent results on the In-The-Wild dataset by reducing Equal Error Rate by 21%.`
  );

  add_model_card(
    model_card_container,
    "RawGAT-ST",
    "Hemlata Tak, Jee-weon Jung, Jose Patino, Madhu Kamble, Massimiliano Todisco, Nicholas Evans",
    "https://arxiv.org/abs/2107.12710",
    "https://github.com/eurecom-asp/RawGAT-ST-antispoofing/tree/main",
    "",
    `Artefacts that serve to distinguish bona fide speech from spoofed or deepfake speech are known to reside in specific subbands and temporal segments. Various approaches can be used to capture and model such artefacts, however, none works well across a spectrum of diverse spoofing attacks. Reliable detection then often depends upon the fusion of multiple detection systems, each tuned to detect different forms of attack. In this paper we show that better performance can be achieved when the fusion is performed within the model itself and when the representation is learned automatically from raw waveform inputs. The principal contribution is a spectro-temporal graph attention network (GAT) which learns the relationship between cues spanning different sub-bands and temporal intervals. Using a model-level graph fusion of spectral (S) and temporal (T) sub-graphs and a graph pooling strategy to improve discrimination, the proposed RawGAT-ST model achieves an equal error rate of 1.06 % for the ASVspoof 2019 logical access database. This is one of the best results reported to date and is reproducible using an open source implementation.`
  );

  const isolation_card_container = document.getElementById(
    "isolation-card-container"
  );

  add_model_card(
    isolation_card_container,
    "Demucs",
    "Simon Rouard, Francisco Massa, Alexandre Défossez",
    "https://arxiv.org/abs/2211.08553",
    "https://github.com/adefossez/demucs?tab=readme-ov-file",
    "Demucs is an open source transformer model for music source separation. Originally maintained Alexandre Défossez originally as an open project while working at meta.",
    "A natural question arising in Music Source Separation (MSS) is whether long range contextual information is useful, or whether local acoustic features are sufficient. In other fields, attention based Transformers have shown their ability to integrate information over long sequences. In this work, we introduce Hybrid Transformer Demucs (HT Demucs), an hybrid temporal/spectral bi-U-Net based on Hybrid Demucs, where the innermost layers are replaced by a cross-domain Transformer Encoder, using self-attention within one domain, and cross-attention across domains. While it performs poorly when trained only on MUSDB, we show that it outperforms Hybrid Demucs (trained on the same data) by 0.45 dB of SDR when using 800 extra training songs. Using sparse attention kernels to extend its receptive field, and per source fine-tuning, we achieve state-of-the-art results on MUSDB with extra training data, with 9.20 dB of SDR. "
  );

  const aixplain_container = (document.getElementById(
    "aixplain-container"
  ).innerHTML = `
          <div class="card border-secondary mb-3">
              <div class="card-body">
                  <h4 class="card-title">aiXplain NoRefER</h4>
                  <h6 class="card-subtitle mb-2 text-muted">
                    ADD AUTHORS
                    <br>
                  </h6>
                  <p class="card-text text-body-secondary">
                    NOT IMPLENTED IN OUR WORK YET
                    </p>
              <h6>Abstract</h6>
              <p class="card-text text-primary-emphasis">
                FIND PAPER
              </p>

              </div>
                <div class="card-footer text-muted">
                  <a href="https://scholar.google.com/" class="card-link">Paper</a>
                  <a href="https://github.com/aixplain/NoRefER" class="card-link">Source</a>
                  <a href="https://huggingface.co/aixplain/NoRefER" class="card-link">Hugging Face</a>
              </div>
            </div>
                `);

  const official_citations_container = document.getElementById(
    "official-citations-container"
  );

  function add_official_citation(title, text) {
    let card_html = `
        <div class="card text-white bg-primary mb-3">
        <div class="card-header">${title}</div>
        <div class="card-body">
            <pre><code class="text-light">
            ${text}
            </pre></code>
        </div>
        </div>
                    `;

    official_citations_container.appendChild(
      new DOMParser().parseFromString(card_html, "text/html").body.firstChild
    );
  }
  add_official_citation(
    "WhisperSpecRNet",
    `
  @inproceedings{kawa23b_interspeech,
      author={Piotr Kawa and Marcin Plata and Michał Czuba and Piotr Szymański and Piotr Syga},
      title={{Improved DeepFake Detection Using Whisper Features}},
      year=2023,
      booktitle={Proc. INTERSPEECH 2023},
      pages={4009--4013},
      doi={10.21437/Interspeech.2023-1537}
  }
  `
  );

  add_official_citation(
    "RawGAT-ST",
    `
  @inproceedings{tak21_asvspoof,
      author={Hemlata Tak and Jee-weon Jung and Jose Patino and Madhu Kamble and Massimiliano Todisco and Nicholas Evans},
      title={{End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection}},
      year=2021,
      booktitle={Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
      pages={1--8},
      doi={10.21437/ASVSPOOF.2021-1}
  }
  `
  );

  add_official_citation(
    "Demucs",
    `
  @inproceedings{rouard2022hybrid,
      title     = {Hybrid Transformers for Music Source Separation},
      author    = {Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
      booktitle = {ICASSP 23},
      year      = {2023}
  }

  @inproceedings{defossez2021hybrid,
      title     = {Hybrid Spectrogram and Waveform Source Separation},
      author    = {D{\'e}fossez, Alexandre},
      booktitle = {Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
      year      = {2021}
  }
  `
  );

  add_official_citation(
    "aiXplain NoRefER",
    `
  @inproceedings{yuksel23_icassp,
      author       = {Kamer Ali Yuksel and Thiago Castro Ferreira and Ahmet Gunduz and Mohamed Al-Badrashiny and Golara Javadi},
      title        = {A Reference-Less Quality Metric for Automatic Speech Recognition via Contrastive-Learning of a Multi-Language Model with Self-Supervision},
      booktitle    = {IEEE International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023},
      pages        = {1--5},
      publisher    = {IEEE},
      year         = {2023},
      url          = {https://doi.org/10.1109/ICASSPW59220.2023.10193003},
      doi          = {10.1109/ICASSPW59220.2023.10193003}
  }
  
  @inproceedings{yuksel23_interspeech,
      author={Kamer Ali Yuksel and Thiago Castro Ferreira and Golara Javadi and Mohamed Al-Badrashiny and Ahmet Gunduz},
      title={NoRefER: a Referenceless Quality Metric for Automatic Speech Recognition via Semi-Supervised Language Model Fine-Tuning with Contrastive Learning},
      year=2023,
      booktitle={Proc. INTERSPEECH 2023},
      pages={466--470},
      doi={10.21437/Interspeech.2023-643}
  }
  
  @inproceedings{javadi2024wordlevel,
      title={Word-Level ASR Quality Estimation for Efficient Corpus Sampling and Post-Editing through Analyzing Attentions of a Reference-Free Metric}, 
      author={Golara Javadi and Kamer Ali Yuksel and Yunsu Kim and Thiago Castro Ferreira and Mohamed Al-Badrashiny},
      eprint={2401.11268},
      archivePrefix={arXiv},
      booktitle    = {IEEE International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2024, Seoul, Korea, April 14-19, 2024},
      publisher    = {IEEE},
      year         = {2024},
      url          = {https://arxiv.org/abs/2401.11268},
      doi          = {10.48550/arXiv.2401.11268}
  }
  `
  );
}
init();
