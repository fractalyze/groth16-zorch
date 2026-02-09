# Copyright 2026 The RabbitSNARK Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for zkey parser."""

import tempfile
from pathlib import Path

from absl.testing import absltest

from rabbitsnark.circom.base import Modulus
from rabbitsnark.circom.zkey import Coefficient, ZKeyV1, parse_zkey
from rabbitsnark.circom.zkey.verifying_key import G1Point, G2Point

# BN254 base field modulus (Fq)
BN254_FQ_MODULUS = (
    21888242871839275222246405745257275088696311157297823662689037894645226208583
)

# BN254 scalar field modulus (Fr)
BN254_FR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


class TestZKeyParser(absltest.TestCase):
    """Tests for zkey file parsing."""

    def setUp(self):
        """Set up test data directory."""
        self.test_data_dir = Path(__file__).parent / "data"

    def test_parse_multiplier_3(self):
        """Test parsing multiplier_3.zkey file."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)

        self.assertEqual(zkey.version, 1)
        self.assertIsInstance(zkey, ZKeyV1)

        # Check header
        self.assertEqual(zkey.header.prover_type, 1)

        # Check Groth16 header
        expected_q = Modulus.from_int(BN254_FQ_MODULUS)
        expected_r = Modulus.from_int(BN254_FR_MODULUS)
        self.assertEqual(zkey.header_groth.q, expected_q)
        self.assertEqual(zkey.header_groth.r, expected_r)
        self.assertEqual(zkey.header_groth.num_vars, 6)
        self.assertEqual(zkey.header_groth.num_public_inputs, 1)
        self.assertEqual(zkey.header_groth.domain_size, 4)

        # Check derived properties
        self.assertEqual(zkey.domain_size, 4)
        self.assertEqual(zkey.num_instance_variables, 2)  # num_public_inputs + 1
        self.assertEqual(zkey.num_witness_variables, 4)  # num_vars - num_public_inputs - 1

    def test_verifying_key(self):
        """Test that verifying key is parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        vkey = zkey.verifying_key

        # Check alpha_g1
        expected_alpha_g1 = G1Point.from_ints(
            x=5700502584084766622350343367608487274977128430049880895783423261700075212785,
            y=9143870410831450591509938003078256759736333300521257694515214164265805259830,
        )
        self.assertEqual(vkey.alpha_g1, expected_alpha_g1)

        # Check beta_g1
        expected_beta_g1 = G1Point.from_ints(
            x=12699714711422499622362310820475830692566951228171954587615996781136226772367,
            y=2601999511749500018822665665362525344184434745926911293241192574303473253831,
        )
        self.assertEqual(vkey.beta_g1, expected_beta_g1)

        # Check beta_g2
        expected_beta_g2 = G2Point.from_ints(
            x=(
                11780196173848324687642894328871430898972567583635494711927265792805257024861,
                3029614260803671687015271480824975868088527303860361358764452805565479529001,
            ),
            y=(
                17817615377642575824268866714659516420384007262298492272608472268977629075434,
                10565581580493997556536063930500447170628763955833078597453719665182760199848,
            ),
        )
        self.assertEqual(vkey.beta_g2, expected_beta_g2)

        # Check gamma_g2
        expected_gamma_g2 = G2Point.from_ints(
            x=(
                10857046999023057135944570762232829481370756359578518086990519993285655852781,
                11559732032986387107991004021392285783925812861821192530917403151452391805634,
            ),
            y=(
                8495653923123431417604973247489272438418190587263600148770280649306958101930,
                4082367875863433681332203403145435568316851327593401208105741076214120093531,
            ),
        )
        self.assertEqual(vkey.gamma_g2, expected_gamma_g2)

        # Check delta_g1
        expected_delta_g1 = G1Point.from_ints(
            x=18121096455458648748006856505340317178704791872899059396361359566439114201168,
            y=1584219057669659447306711278235088033786171030532185363250775914928871374123,
        )
        self.assertEqual(vkey.delta_g1, expected_delta_g1)

        # Check delta_g2
        expected_delta_g2 = G2Point.from_ints(
            x=(
                12202969132968262321607709426694195568617274504067880373401633817598633451917,
                3518609536734967363313514083213104545803320919611827958997148133314959258666,
            ),
            y=(
                18833075355260917945174052299300691737626083696241282320643937004218902395077,
                9205653431456404075288921182558898955030763586613450941955464042050735041036,
            ),
        )
        self.assertEqual(vkey.delta_g2, expected_delta_g2)

    def test_ic_points(self):
        """Test that IC points are parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        expected_ic = [
            G1Point.from_ints(
                x=1400989341879513116647759947859271187117391672677487101192308885590924596480,
                y=18827163924960691750679623127657074266908067481725903803154895122477837234033,
            ),
            G1Point.from_ints(
                x=21594466749489205217527764338982336503042047314074010288653645878163201627935,
                y=12389944772204356478767528982065842735667468360799583461061670022670871632383,
            ),
        ]
        self.assertLen(zkey.ic, 2)
        self.assertEqual(zkey.ic, expected_ic)

    def test_coefficients(self):
        """Test that coefficients are parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        # Expected coefficients from the C++ test
        expected = [
            Coefficient.from_ints(
                matrix=0,
                constraint=0,
                signal=2,
                value=21888242871839275222246405745257275088548364400416034343698204186575808495616,
            ),
            Coefficient.from_ints(matrix=1, constraint=0, signal=3, value=1),
            Coefficient.from_ints(
                matrix=0,
                constraint=1,
                signal=5,
                value=21888242871839275222246405745257275088548364400416034343698204186575808495616,
            ),
            Coefficient.from_ints(matrix=1, constraint=1, signal=4, value=1),
            Coefficient.from_ints(matrix=0, constraint=2, signal=0, value=1),
            Coefficient.from_ints(matrix=0, constraint=3, signal=1, value=1),
        ]

        self.assertLen(zkey.coefficients, 6)
        self.assertEqual(zkey.coefficients, expected)

    def test_points_a1(self):
        """Test that points_a1 are parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        expected_points_a1 = [
            G1Point.from_ints(
                x=8858563469144920540528478490224638442973773873152551307670564100347093499191,
                y=7888214391937843930525848128254405915157714572978190674521564636068162216311,
            ),
            G1Point.from_ints(
                x=14537214592124271965353533016257772100455033778428577041971202446686849252644,
                y=2198766467867023896703420308951432042782623727887618971273865174145643356495,
            ),
            G1Point.from_ints(
                x=8437302598248383817148383036741547214048558400312301295747047351838256772123,
                y=4253086419746464003785043685439509391398040483296248505707498714848332192725,
            ),
            G1Point.from_ints(x=0, y=0),
            G1Point.from_ints(x=0, y=0),
            G1Point.from_ints(
                x=18141870587741836486360437684811661514896911334995841933942081072546739652377,
                y=11898889550822544273094627075076607374273361105699305622414170117806818640166,
            ),
        ]
        self.assertLen(zkey.points_a1, 6)
        self.assertEqual(zkey.points_a1, expected_points_a1)

    def test_points_b1(self):
        """Test that points_b1 are parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        expected_points_b1 = [
            G1Point.from_ints(x=0, y=0),
            G1Point.from_ints(x=0, y=0),
            G1Point.from_ints(x=0, y=0),
            G1Point.from_ints(
                x=8437302598248383817148383036741547214048558400312301295747047351838256772123,
                y=17635156452092811218461362059817765697298270674001575156981539179796894015858,
            ),
            G1Point.from_ints(
                x=18141870587741836486360437684811661514896911334995841933942081072546739652377,
                y=9989353321016730949151778670180667714422950051598518040274867776838407568417,
            ),
            G1Point.from_ints(x=0, y=0),
        ]
        self.assertLen(zkey.points_b1, 6)
        self.assertEqual(zkey.points_b1, expected_points_b1)

    def test_points_b2(self):
        """Test that points_b2 are parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        expected_points_b2 = [
            G2Point.from_ints(x=(0, 0), y=(0, 0)),
            G2Point.from_ints(x=(0, 0), y=(0, 0)),
            G2Point.from_ints(x=(0, 0), y=(0, 0)),
            G2Point.from_ints(
                x=(
                    11802355142842158477844840276643950524651646845857959153292587217381565327694,
                    457802140950752837652486610695137856486735570684796633854607481168528542690,
                ),
                y=(
                    20890171563279906566473411100997246109253767120687561732766386895834736963353,
                    16651273739129079927357167917012486078894432847976493185750898267875291365102,
                ),
            ),
            G2Point.from_ints(
                x=(
                    10497384263581811947331014280742114350358633905325456855203962488034692371918,
                    18830443503724104126054724199526976415213768169570049829023119362975529483862,
                ),
                y=(
                    2897811547300447900653975040323384022566235978103493011133012417217802784890,
                    16901016173912215869936744693239612773536725857766261524800318448291024785326,
                ),
            ),
            G2Point.from_ints(x=(0, 0), y=(0, 0)),
        ]
        self.assertLen(zkey.points_b2, 6)
        self.assertEqual(zkey.points_b2, expected_points_b2)

    def test_points_c1(self):
        """Test that points_c1 are parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        expected_points_c1 = [
            G1Point.from_ints(
                x=6484623682921116324480150495004051666793989100055782957602702403603681644087,
                y=9929291865986144258563515295092942842219006739962304434423595963929152482511,
            ),
            G1Point.from_ints(
                x=2428323497801148585616314929247100239345590148163609103036877009446204389916,
                y=1369420932117767430108173316323061012699395351388826295491040858566319703347,
            ),
            G1Point.from_ints(
                x=13708611801147211497962477788407562751837989814094039031176550253520745906453,
                y=18151634040197625351146955912693877471406025170780331525940368525891263003135,
            ),
            G1Point.from_ints(
                x=18284461151151484053721771333035891989697029546383442693996002288554315091425,
                y=16368214353654036882062732785064467942017556803959036709508171804841339585293,
            ),
        ]
        # num_vars - num_public_inputs - 1 = 6 - 1 - 1 = 4
        self.assertLen(zkey.points_c1, 4)
        self.assertEqual(zkey.points_c1, expected_points_c1)

    def test_points_h1(self):
        """Test that points_h1 are parsed correctly."""
        zkey_path = self.test_data_dir / "multiplier_3.zkey"
        zkey = parse_zkey(zkey_path)
        self.assertIsInstance(zkey, ZKeyV1)

        expected_points_h1 = [
            G1Point.from_ints(
                x=14491578304961494983864577860005130648083128241617887000567655109826889517765,
                y=747219666158167233177659053461241136834613714217214833065101750371763762297,
            ),
            G1Point.from_ints(
                x=18146981344323077863601561933187745037217273647724943548414954460732389867565,
                y=12817730263653070328990536695886808120537471796272335985833677017167603721628,
            ),
            G1Point.from_ints(
                x=4366845324910879958335909021287873980685276473035323564144861870453367362588,
                y=15317533781337899355741367900150169453079806490427679757823792291054041331892,
            ),
            G1Point.from_ints(
                x=2710629677537908437939962149843601091625518295461558128309313181856887222076,
                y=5586033426926327725348881354712158558954133315341909946743947206034841045334,
            ),
        ]
        # domain_size = 4
        self.assertLen(zkey.points_h1, 4)
        self.assertEqual(zkey.points_h1, expected_points_h1)

    def test_parse_invalid_magic(self):
        """Test that invalid magic raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_file = Path(tmp_dir) / "invalid.zkey"
            invalid_file.write_bytes(b"xxxx")

            with self.assertRaisesRegex(ValueError, "Invalid magic"):
                parse_zkey(invalid_file)

    def test_parse_unsupported_version(self):
        """Test that unsupported version raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_file = Path(tmp_dir) / "invalid.zkey"
            # Magic + version 99
            invalid_file.write_bytes(b"zkey" + (99).to_bytes(4, "little"))

            with self.assertRaisesRegex(ValueError, "Unsupported version"):
                parse_zkey(invalid_file)


if __name__ == "__main__":
    absltest.main()
