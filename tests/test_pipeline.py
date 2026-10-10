import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from pipeline.ocr_state import normalize_plate, get_prefix, get_state
from pipeline.plate_detect import detect_plate_info, _tokens_to_plate, vehicle_type_from_path
from pipeline.preprocess import resize_long_edge


def token(text, confidence=0.9, x=0, y=0):
    return ([[x, y], [x + 30, y], [x + 30, y + 10], [x, y + 10]], text, confidence)


class MappingTests(unittest.TestCase):
    def test_normalizes_spaces_and_punctuation(self):
        self.assertEqual(normalize_plate(' wsp 5591! '), 'WSP5591')
        self.assertEqual(normalize_plate(''), '')

    def test_first_letter_mapping(self):
        self.assertEqual(get_prefix('ABF7429'), 'A')
        self.assertEqual(get_state('A'), 'Ipoh')
        self.assertEqual(get_state('W'), 'Kuala Lumpur')
        self.assertEqual(get_state('P'), 'Penang')

    def test_unknown_prefix_remains_unknown(self):
        self.assertEqual(get_state(get_prefix('ZZZ123')), 'Unknown')
        self.assertEqual(get_prefix('123'), 'Unknown')

    def test_vehicle_label_uses_folder_not_image_content(self):
        self.assertEqual(vehicle_type_from_path(r'C:\images\Truck\sample.jpg'), 'Truck')
        self.assertEqual(vehicle_type_from_path('/images/motorcycle/sample.jpg'), 'Motorcycle')
        self.assertEqual(vehicle_type_from_path('/images/Bus/sample.jpg'), 'Bus')
        self.assertEqual(vehicle_type_from_path('/photos/sample.jpg'), 'Unknown')


class TokenTests(unittest.TestCase):
    def test_joins_horizontal_tokens_by_x_position(self):
        plate, _, _ = _tokens_to_plate([token('5591', x=60), token('WSP', x=0)])
        self.assertEqual(plate, 'WSP5591')

    def test_joins_two_rows(self):
        plate, _, _ = _tokens_to_plate([token('5591', x=0, y=50), token('WSP', x=40)])
        self.assertEqual(plate, 'WSP5591')

    def test_rejects_unrelated_text(self):
        self.assertEqual(_tokens_to_plate([token('DEALERSHIP')])[0], '')
        self.assertEqual(_tokens_to_plate([])[0], '')


class ImagePipelineTests(unittest.TestCase):
    def test_missing_and_corrupt_files_return_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.jpg'
            self.assertEqual(detect_plate_info(str(missing)).mode, 'ERROR')
            corrupt = Path(directory) / 'corrupt.jpg'
            corrupt.write_bytes(b'not an image')
            self.assertEqual(detect_plate_info(str(corrupt)).mode, 'ERROR')
            corrupt.write_bytes(b'')
            self.assertEqual(detect_plate_info(str(corrupt)).mode, 'ERROR')

    def test_unicode_image_path_and_candidate_result(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'plate_测试.png'
            success, encoded = cv2.imencode('.png', np.zeros((100, 300, 3), dtype=np.uint8))
            self.assertTrue(success)
            encoded.tofile(str(image_path))
            with patch('pipeline.plate_detect._find_plate_candidates', return_value=[(1.0, (10, 10, 150, 30))]), patch('pipeline.plate_detect._easyocr_read', return_value=[token('WSP5591')]) as ocr:
                result = detect_plate_info(str(image_path))
            self.assertEqual(result.plate_text, 'WSP5591')
            self.assertEqual(result.state, 'Kuala Lumpur')
            self.assertIn('candidate=1/1', result.debug)
            ocr.assert_called_once()

    def test_full_image_fallback_and_no_match(self):
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        with patch('pipeline.plate_detect._imread_unicode', return_value=image), patch('pipeline.plate_detect._find_plate_candidates', return_value=[]), patch('pipeline.plate_detect._easyocr_read', return_value=[token('PFQ5217')]):
            result = detect_plate_info('/images/van/example.jpg')
        self.assertEqual(result.state, 'Penang')
        self.assertEqual(result.vehicle_type, 'Van')
        self.assertIn('fallback_full_easyocr', result.debug)
        with patch('pipeline.plate_detect._imread_unicode', return_value=image), patch('pipeline.plate_detect._find_plate_candidates', return_value=[]), patch('pipeline.plate_detect._easyocr_read', return_value=[]):
            result = detect_plate_info('example.jpg')
        self.assertEqual(result.plate_text, '')
        self.assertEqual(result.quality, -1.0)

    def test_resize_preserves_aspect_ratio(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertEqual(resize_long_edge(image, 100).shape, (50, 100, 3))


if __name__ == '__main__':
    unittest.main()
