"""Testes do pipeline (processor.py) — sem rede, sem ffmpeg, sem tocar no _cache real.

Rodar da pasta da ferramenta:  python -m unittest discover -s tests
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import processor as p  # noqa: E402

YT = "https://youtu.be/LSZCkC0k93s"


class SourceOptsTest(unittest.TestCase):
    def test_youtube_usa_client_android_e_formato_muxado(self):
        o = p._source_ydl_opts(YT)
        self.assertEqual(o["extractor_args"], {"youtube": {"player_client": ["android"]}})
        self.assertEqual(o["format"], "b[protocol=https][acodec!=none][vcodec!=none]/18/b")

    def test_nao_youtube_nao_forca_formato(self):
        self.assertNotIn("format", p._source_ydl_opts("https://www.instagram.com/reel/x/"))


class ProcessTest(unittest.TestCase):
    """process() ponta a ponta com download/audio/transcricao falsos e _cache isolado."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_test_")
        cache = os.path.join(self.tmp, "_cache")
        for name, val in (("CACHE_DIR", cache),
                          ("SEEN_PATH", os.path.join(cache, "_seen.json")),
                          ("TIMINGS_PATH", os.path.join(cache, "_timings.json"))):
            patcher = mock.patch.object(p, name, val)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _fake_download(self, size, info_extra=None):
        def dl(source, workdir, progress=None):
            path = os.path.join(workdir, "source.mp4")
            with open(path, "wb") as f:
                f.write(b"\0" * size)
            return path, {"title": "Video", "source": source, "chapters": None,
                          "duration": 267, "description": None, **(info_extra or {})}
        return dl

    def _run(self, size, audio_secs, info_extra=None, **kw):
        transcript = {"text": "ola", "segments": [{"start": 0, "end": 1, "text": "ola"}],
                      "words": [], "engine": "groq:x"}
        with mock.patch.object(p, "download_or_locate", self._fake_download(size, info_extra)), \
             mock.patch.object(p, "extract_audio", lambda v, w: os.path.join(w, "audio.ogg")), \
             mock.patch.object(p, "get_duration", lambda path: audio_secs), \
             mock.patch.object(p, "transcribe", mock.Mock(return_value=transcript)) as tr:
            res = p.process(YT, max_screenshots=0, cache_id="t", force=True, **kw)
        return res, tr

    def test_download_completo_transcreve_o_audio(self):
        res, tr = self._run(size=9_000_000, audio_secs=267.0,
                            info_extra={"filesize": 9_700_000})
        tr.assert_called_once()
        self.assertEqual(res["metadata"]["engine"], "groq:x")
        self.assertEqual(res["segments"][0]["text"], "ola")
        self.assertTrue(os.path.isfile(p.load_meta("t")["video_path"]))

    def _json3_response(self):
        body = (b'{"events": [{"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "bom"},'
                b' {"utf8": " dia"}]}, {"tStartMs": 1500, "segs": [{"utf8": "\\n"}]}]}')
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.read.return_value = body
        return resp

    def test_truncado_por_tamanho_usa_legendas_sem_transcrever(self):
        # caso real: 109 KB de ~9.7 MB (filesize_approx), formato 18 do YouTube
        with mock.patch("urllib.request.urlopen", return_value=self._json3_response()) as uo:
            res, tr = self._run(size=109_247, audio_secs=267.0, info_extra={
                "filesize": 9_748_950, "caption": ("pt-orig", "https://yt/cap.json3")})
        uo.assert_called_once()
        tr.assert_not_called()
        self.assertEqual(res["segments"], [{"start": 0.0, "end": 2.0, "text": "bom dia"}])
        self.assertIn("legendas do YouTube (pt-orig)", res["metadata"]["engine"])
        self.assertEqual(res["frames"], [])
        self.assertIsNone(p.load_meta("t")["video_path"])

    def test_truncado_pela_duracao_do_audio_sem_legenda_erro_claro(self):
        # sem filesize na info: so o audio (347 bytes -> 0 s de 267 s) denuncia
        with self.assertRaises(p.TruncatedDownloadError) as ctx:
            self._run(size=109_247, audio_secs=0.0, info_extra={"caption": None})
        self.assertIn("INCOMPLETO", str(ctx.exception))
        self.assertIn("0.0 s de audio de ~267.0", str(ctx.exception))

    def test_legenda_que_falha_ao_baixar_vira_erro_claro(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("403")), \
             self.assertRaises(p.TruncatedDownloadError):
            self._run(size=109_247, audio_secs=267.0, info_extra={
                "filesize": 9_748_950, "caption": ("pt-orig", "https://yt/cap.json3")})


class TruncationHelpersTest(unittest.TestCase):
    def test_nota_so_abaixo_do_limiar(self):
        self.assertIsNone(p._truncation_note(8.0, 9.7, "MB"))      # approx pode errar
        self.assertIsNone(p._truncation_note(0.1, None, "MB"))     # sem referencia
        self.assertEqual(p._truncation_note(0.1, 9.7, "MB"), "0.1 MB de ~9.7 MB esperados")

    def test_escolhe_legenda_original_no_idioma(self):
        j = [{"ext": "vtt", "url": "v"}, {"ext": "json3", "url": "j"}]
        info = {"language": "pt", "subtitles": {},
                "automatic_captions": {"en": j, "pt": j, "pt-orig": j}}
        self.assertEqual(p._pick_caption(info), ("pt-orig", "j"))
        self.assertIsNone(p._pick_caption({"language": "pt"}))


if __name__ == "__main__":
    unittest.main()
