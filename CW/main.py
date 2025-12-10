import argparse
import prototype_one
import train
import model_test

def main():
    parser = argparse.ArgumentParser()

    # mode choice
    parser.add_argument(
        "--mode",
        required=True,
        choices=["generate_data", "train_classifier", "test_planner"]
    )

    # train_classifier parameter
    parser.add_argument("--data", default="/Users/thun/Downloads/cw_latest_ver/three_finger_cylinder.csv")
    parser.add_argument("--out-train", default="models")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv", type=int, default=5)

    # test_planner parameter
    parser.add_argument("--model", type=str, default="models/best_model.pkl", help="Path to saved model")
    parser.add_argument("--out", type=str, default="eval", help="Output directory for results")
    parser.add_argument("--samples", type=int, default=10, help="Number of grasp attempts")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--radius", type=float, default=0.7, help="Sampling sphere radius (reduced default)")
    parser.add_argument("--gripper", type=str, default="three-finger", choices=["two-finger", "three-finger"], help="Gripper type")
    parser.add_argument("--object", type=str, default="cube", choices=["cube", "cylinder"], help="Object type")
    parser.add_argument("--flip-offset", action="store_true", help="Flip three-finger orientation offset (pitch sign)")
    parser.add_argument("--contact-thresh", type=float, default=0.65, help="Contact ratio threshold for success")
    parser.add_argument("--friction", type=float, default=1.3, help="Lateral friction when enabled")
    parser.add_argument("--debug-contacts", action="store_true", help="Print contact counts during lift")

    # generate_data parameter
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100
    )

    args = parser.parse_args()

    if args.mode == "generate_data":
        prototype_one.main(args.num_samples)

    elif args.mode == "train_classifier":
        train.main(
            data=args.data,
            out=args.out_train,
            test_size=args.test_size,
            cv=args.cv,
        )

    elif args.mode == "test_planner":
        model_test.main(
        model=args.model,
        out=args.out,
        samples=args.num_samples,
        gui=args.gui,
        radius=args.radius,
        gripper=args.gripper,
        object=args.object,
        flip_offset=args.flip_offset,
        contact_thresh=args.contact_thresh,
        friction=args.friction,
        debug_contacts=args.debug_contacts,
    )

if __name__ == "__main__":
    main()
